import numpy as np
from mpi4py import MPI
import torch


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    #TODO: Your code here
    mp_idx = rank % mp_size
    dp_idx = rank // mp_size
    mp_comm = comm.Split(color=dp_idx, key=mp_idx)
    dp_comm = comm.Split(color=mp_idx, key=dp_idx)

    part_in_dim = in_dim
    part_out_dim = out_dim

    # W is of shape [in_dim, out_dim]
    if fc_layer in ('fc_q', 'fc_k', 'fc_v'):
        part_out_dim = out_dim // mp_size
        part_in_dim = in_dim 
    elif fc_layer == 'fc_o':
        part_out_dim = out_dim
        part_in_dim = in_dim // mp_size

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim

def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    stride_seq = x.shape[2]
    stride_batch = x.shape[1] * stride_seq
    stride_blk = x.shape[0] * stride_batch

    collected_x = np.empty((x.shape[0], x.shape[1], x.shape[2] * mp_size))

    x_flat = x.reshape(-1)
    collected_x_flat = np.empty((stride_blk*mp_size))
    mp_comm.Allgather(x_flat, collected_x_flat)
    
    
    for blk in range(mp_size):
        for batch in range(x.shape[0]):
            for seq in range(x.shape[1]):
                dim_3_start = blk * stride_seq
                flat_start = blk*stride_blk + batch * stride_batch + seq * stride_seq
                collected_x[batch, seq, dim_3_start: dim_3_start + stride_seq] = collected_x_flat[flat_start : flat_start+stride_seq]
   
    return collected_x


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    stride_seq = out.shape[2]
    stride_batch = out.shape[1] * stride_seq
    stride_blk = out.shape[0] * stride_batch

    collected_out = np.empty((out.shape[0], out.shape[1], out.shape[2] * mp_size))

    out_flat = out.reshape(-1)
    collected_out_flat = np.empty((out.shape[0]*out.shape[1]*out.shape[2]*mp_size))
    mp_comm.Allgather(out_flat, collected_out_flat)


    for blk in range(mp_size):
        for batch in range(out.shape[0]):
            for seq in range(out.shape[1]):
                dim_3_start = blk * stride_seq
                flat_start = blk*stride_blk + batch * stride_batch + seq * stride_seq
                collected_out[batch, seq, dim_3_start : dim_3_start+stride_seq] = collected_out_flat[flat_start : flat_start+stride_seq] 

    return collected_out

def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.
    
    In our setup, the full output_grad is a 3-D tensor of shape 
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.
    
    Parameters
    ----------
    output_grad : np.ndarray
        The full output gradient from fc_o with shape 
        (batch_size, seq_length, out_dim).
    mp_group_idx : int
        The current model parallel node's index.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_output_grad : np.ndarray
        The local output gradient for this MP node with shape 
        (batch_size, seq_length, out_dim // mp_size).
    """
    chunks = np.array_split(output_grad, mp_size, axis=2)
    collected_output_grad = chunks[mp_group_idx]
    return collected_output_grad

def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Use reduce-scatter / all-to-all to combine the contributions for grad_x from all nodes
    and scatter the reduced result along the input feature dimension.
    
    The grad_x tensor (gradient with respect to fc_o's input) has shape
        (batch_size, seq_length, in_dim),
    and the fc_o's weight matrix is sharded along the in_dim axis. In the 
    backward pass, each node computes a local grad_x and then these must be 
    summed across nodes. Instead of summing the full tensor and then slicing,
    we perform a reduce-scatter / all-to-all.
    
    Parameters
    ----------
    grad_x : np.ndarray
        The locally computed grad_x for fc_o, of shape 
        (batch_size, seq_length, in_dim).
    mp_comm :
        The model parallel communicator. It is assumed to expose methods such as reduce-scatter / all-to-all.
    mp_size : int
        The total number of model parallel nodes.
    
    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and scattered grad_x with shape 
        (batch_size, seq_length, in_dim // mp_size).
    """
    batch_size = grad_x.shape[0]
    collected_grad_x = np.empty((batch_size, grad_x.shape[1], grad_x.shape[2] // mp_size))
    for batch in range(batch_size):
        for seq in range(grad_x.shape[1]):
            # Reduce-scatter along the last dimension (in_dim)
            mp_comm.Reduce_scatter(grad_x[batch, seq, :], collected_grad_x[batch, seq, :], op=MPI.SUM)
    return collected_grad_x