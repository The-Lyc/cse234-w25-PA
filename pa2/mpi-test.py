from mpi4py import MPI
from mpi_wrapper import Communicator
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_case",
    type=str,
    help="MPI names for different toy examples",
    default="",
    choices=["allreduce", "allgather", "reduce_scatter", "split", 'alltoall', 'myallreduce', 'myalltoall'],
)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    comm = Communicator(comm)

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    args = parser.parse_args()

    if args.test_case == "allreduce":
        """
        Allreduce example
        """

        r = np.random.randint(0, 100, 100)
        rr = np.empty(100, dtype=int)

        print("Rank " + str(rank) + ": " + str(r))

        comm.Barrier()
        comm.Allreduce(r, rr, op=MPI.MIN)

        if rank == 0:
            print("Allreduce: " + str(rr))
    
    if args.test_case == "myallreduce":

        num_runs = 100
        len_array = 100

        # Lists to store the execution time for each run
        allreduce_times = []
        myallreduce_times = []

        # Flag to record if any run has an incorrect result.
        all_runs_correct = True

        for run in range(num_runs):
            # Create a random integer array (each process gets its own random array)
            r = np.random.randint(0, 100, len_array)

            # Arrays to store the reduction results
            rr_allreduce   = np.empty(len_array, dtype=int)
            rr_myallreduce = np.empty(len_array, dtype=int)

            # --- Built-in Allreduce Timing ---
            comm.Barrier()  # Synchronize all processes before timing
            start = MPI.Wtime()
            comm.Allreduce(r, rr_allreduce, op=MPI.MIN)
            comm.Barrier()  # Synchronize after call
            elapsed_all = MPI.Wtime() - start
            allreduce_times.append(elapsed_all)

            # --- Custom myAllreduce Timing ---
            comm.Barrier()  # Synchronize all processes before timing
            start = MPI.Wtime()
            comm.myAllreduce(r, rr_myallreduce, op=MPI.MIN)
            comm.Barrier()  # Synchronize after call
            elapsed_my = MPI.Wtime() - start
            myallreduce_times.append(elapsed_my)

            # --- Check Correctness ---
            if not np.array_equal(rr_allreduce, rr_myallreduce):
                all_runs_correct = False
                print("Rank {}: Run {}: ERROR: myAllreduce result does not match Allreduce".format(rank, run))
                # Optionally, print the differing arrays for debugging:
                if rank == 0:
                    print("Allreduce result:   ", rr_allreduce)
                    print("myAllreduce result: ", rr_myallreduce)
            else:
                if rank == 0:
                    print("Run {}: Correct results.".format(run))

        # Compute the average execution times over all runs.
        avg_allreduce = sum(allreduce_times) / num_runs
        avg_myallreduce = sum(myallreduce_times) / num_runs

        # Only rank 0 prints the summary.
        if rank == 0:
            print("\nSummary over {} runs:".format(num_runs))
            if all_runs_correct:
                print("All runs produced correct results.")
            else:
                print("Some runs produced incorrect results!")
            print("Average MPI.Allreduce time: {:.6f} seconds".format(avg_allreduce))
            print("Average myAllreduce time:   {:.6f} seconds".format(avg_myallreduce))
    
    elif args.test_case == "allgather":
        """
        Allgather example
        """

        r = np.random.randint(0, 100, 2)
        rr = np.empty(16, dtype=int)

        print("Rank " + str(rank) + ": " + str(r))

        comm.Barrier()
        comm.Allgather(r, rr)

        if rank == 0:
            print("Allgather: " + str(rr))

    elif args.test_case == "reduce_scatter":
        """
        Reduce_scatter example
        """

        r = np.random.randint(0, 100, 16)
        rr = np.empty(2, dtype=int)

        print("Rank " + str(rank) + ": " + str(r))

        comm.Barrier()
        comm.Reduce_scatter(r, rr, op=MPI.MIN)

        print("Rank " + str(rank) + " After Reduce_scatter: " + str(rr))

    elif args.test_case == "split":
        """
        Split example (group-wise reduce)
        split into 4 groups based on the modulo operation:
         Group 0: (0, 4)
         Group 1: (1, 5)
         Group 2: (2, 6)
         Group 3: (3, 7)
        """

        r = np.random.randint(0, 100, 10)
        rr = np.empty(10, dtype=int)

        print("Rank " + str(rank) + ": " + str(r))

        key = rank
        color = rank % 4

        group_comm = comm.Split(key=key, color=color)

        group_comm.Barrier()
        group_comm.Allreduce(r, rr, op=MPI.MIN)

        print("Rank " + str(rank) + " After split and Allreduce: " + str(rr))

    elif args.test_case == "alltoall":
        """
        All-to-all example
        Each process creates a send buffer of size nprocs.
        The element at index i is set to a unique value (e.g., rank*100 + i)
        so that after the all-to-all, every process receives one element from each rank.
        """
        # Create a send buffer with one element for each process.
        send_data = np.empty(nprocs, dtype=int)
        for i in range(nprocs):
            send_data[i] = rank * 100 + i  # Unique value for demonstration

        # Prepare a receive buffer of the same size.
        recv_data = np.empty(nprocs, dtype=int)

        print("Rank " + str(rank) + " sending: " + str(send_data))

        comm.Barrier()
        comm.Alltoall(send_data, recv_data)

        print("Rank " + str(rank) + " received: " + str(recv_data))

    elif args.test_case == "myalltoall":

        nprocs = comm.Get_size()
        num_runs = 100

        # Lists to store the execution times for each run
        alltoall_times = []
        myalltoall_times = []
        all_runs_correct = True

        for run in range(num_runs):
            # --- Prepare Data ---
            # Each process creates a send buffer of size nprocs.
            # The element at index i is unique: rank*100 + i.
            send_data = np.empty(nprocs, dtype=int)
            for i in range(nprocs):
                send_data[i] = rank * 100 + i

            # Create receive buffers for both methods.
            recv_data_alltoall = np.empty(nprocs, dtype=int)
            recv_data_myalltoall = np.empty(nprocs, dtype=int)

            # --- Built-in MPI Alltoall Timing ---
            comm.Barrier()  # Synchronize all processes
            start = MPI.Wtime()
            comm.Alltoall(send_data, recv_data_alltoall)
            comm.Barrier()  # Ensure all processes complete the call
            elapsed_all = MPI.Wtime() - start
            alltoall_times.append(elapsed_all)

            # --- Custom myAlltoall Timing ---
            comm.Barrier()  # Synchronize before starting the custom call
            start = MPI.Wtime()
            comm.myAlltoall(send_data, recv_data_myalltoall)
            comm.Barrier()  # Synchronize after the custom call
            elapsed_custom = MPI.Wtime() - start
            myalltoall_times.append(elapsed_custom)

            # --- Check Correctness ---
            if not np.array_equal(recv_data_alltoall, recv_data_myalltoall):
                all_runs_correct = False
                print("Rank {}: Run {}: ERROR: myAlltoall result does not match MPI.Alltoall".format(rank, run))
                if rank == 0:
                    print("MPI.Alltoall result:  ", recv_data_alltoall)
                    print("myAlltoall result:    ", recv_data_myalltoall)
            else:
                if rank == 0:
                    print("Run {}: Correct results.".format(run))

        # Compute average times over all runs.
        avg_alltoall = sum(alltoall_times) / num_runs
        avg_myalltoall = sum(myalltoall_times) / num_runs

        # Only rank 0 prints the summary.
        if rank == 0:
            print("\nSummary over {} runs:".format(num_runs))
            if all_runs_correct:
                print("All runs produced correct results.")
            else:
                print("Some runs produced incorrect results!")
            print("Average MPI.Alltoall time: {:.6f} seconds".format(avg_alltoall))
            print("Average myAlltoall time:   {:.6f} seconds".format(avg_myalltoall))
    else:
        print(f"This is rank {rank}.")
