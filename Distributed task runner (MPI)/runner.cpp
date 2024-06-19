#include "runner.hpp"
#include <fstream>
#include <queue>
#include <algorithm>

/*
 * ==========================================
 * | [START] OK TO MODIFY THIS FILE [START] |
 * ==========================================
 */
// MPI message definitions for our task
constexpr int task_t_num_blocks = 8;
constexpr int task_t_lengths[task_t_num_blocks] = {1, 1, 1, 1, 1, 1, 4, 4};
constexpr MPI_Aint task_t_displs[task_t_num_blocks] = {
  offsetof(struct task_t, id),
  offsetof(struct task_t, gen),
  offsetof(struct task_t, type),
  offsetof(struct task_t, arg_seed),
  offsetof(struct task_t, output),
  offsetof(struct task_t, num_dependencies),
  offsetof(struct task_t, dependencies),
  offsetof(struct task_t, masks),
};
inline const MPI_Datatype task_t_types[task_t_num_blocks] = {
  MPI_UINT32_T, MPI_INT, MPI_INT, MPI_UINT32_T, MPI_UINT32_T, MPI_INT, MPI_UINT32_T, MPI_UINT32_T
};

void run_seq(int rank, int num_procs, metric_t& stats, params_t& params);

void run_all_tasks(int rank, int num_procs, metric_t &stats, params_t &params) {

  MPI_Datatype MPI_TASK_T;
  MPI_Type_create_struct(task_t_num_blocks, task_t_lengths, task_t_displs, task_t_types, &MPI_TASK_T);
  MPI_Type_commit(&MPI_TASK_T);

  MPI_Status stat;

  if (num_procs == 1) {
    run_seq(rank, num_procs, stats, params);

  } else if (rank == 0) {
    std::queue<task_t> task_queue;

    std::ifstream istrm(params.input_path, std::ios::binary);
    // Read initial tasks
    int count;
    istrm >> count;

    for (int i = 0; i < count; ++i) {
      task_t task;
      int type;
      istrm >> type >> task.arg_seed;
      task.type = static_cast<TaskType>(type);
      task.id = task.arg_seed;
      task.gen = 0;
      task_queue.push(task);
    }

    // Keep track of processes that are idle
    std::queue<int> free_processes;
    for (int i = 1; i != num_procs; ++i)
      free_processes.push(i);

    int num_new_tasks = 0;
    int num_tasks_executing = 0;
    std::vector<task_t> task_buffer(Nmax);

    // Send initial batch of tasks
    int end = std::min(task_queue.size(), free_processes.size());
    for (int i = 0; i != end; ++i)
    {
      int dest_rank = free_processes.front();
      int dest_tag = dest_rank;
      free_processes.pop();

      MPI_Send(&(task_queue.front()), 1, MPI_TASK_T, dest_rank, dest_tag, MPI_COMM_WORLD);
      task_queue.pop();

      ++num_tasks_executing;
    }

    while (true)
    {
      if (task_queue.empty() && num_tasks_executing == 0) break;

      // Receive task completion results
      MPI_Recv(&num_new_tasks, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
      int source_rank = stat.MPI_SOURCE;
      free_processes.push(source_rank);
      --num_tasks_executing;
      
      if (num_new_tasks != 0)
      {
        MPI_Recv(task_buffer.data(), num_new_tasks, MPI_TASK_T, source_rank, 0, MPI_COMM_WORLD, &stat);
        for (int i = 0; i != num_new_tasks; ++i)
          task_queue.push(task_buffer[i]);
      }

      // Send new tasks in reaction to task completion
      int end = std::min(task_queue.size(), free_processes.size());
      for (int i = 0; i != end; ++i)
      {
        int dest_rank = free_processes.front();
        int dest_tag = dest_rank;
        free_processes.pop();

        MPI_Send(&(task_queue.front()), 1, MPI_TASK_T, dest_rank, dest_tag, MPI_COMM_WORLD);
        task_queue.pop();

        ++num_tasks_executing;
      }
    }

    // Send termination task to processes
    task_t terminate;
    terminate.gen = -1;
    for (int i = 1; i != num_procs; ++i) {
      MPI_Send(&terminate, 1, MPI_TASK_T, i, i, MPI_COMM_WORLD);
    }

  } else {
    int num_new_tasks = 0;
    std::vector<task_t> task_buffer(Nmax);
    task_t curr_task;
  
    while (true) {
      MPI_Recv(&curr_task, 1, MPI_TASK_T, 0, rank, MPI_COMM_WORLD, &stat);

      // If received termination task
      if (curr_task.gen == -1) {
        break;
      }

      execute_task(stats, curr_task, num_new_tasks, task_buffer);

      MPI_Send(&num_new_tasks, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      if (num_new_tasks != 0)
        MPI_Send(task_buffer.data(), num_new_tasks, MPI_TASK_T, 0, 0, MPI_COMM_WORLD);
    }
  }
}

void run_all_tasks_naive(int rank, int num_procs, metric_t &stats, params_t &params) {

  MPI_Datatype MPI_TASK_T;
  MPI_Type_create_struct(task_t_num_blocks, task_t_lengths, task_t_displs, task_t_types, &MPI_TASK_T);
  MPI_Type_commit(&MPI_TASK_T);

  MPI_Status stat;

  if (rank == 0) {
    std::queue<task_t> task_queue;

    std::ifstream istrm(params.input_path, std::ios::binary);
    // Read initial tasks
    int count;
    istrm >> count;

    for (int i = 0; i < count; ++i) {
      task_t task;
      int type;
      istrm >> type >> task.arg_seed;
      task.type = static_cast<TaskType>(type);
      task.id = task.arg_seed;
      task.gen = 0;
      task_queue.push(task);
    }

    int num_new_tasks = 0;
    std::vector<task_t> task_buffer(Nmax);

    while (true) {

      if (task_queue.empty()) {
        break;
      }
           
      int end = std::min(static_cast<int>(task_queue.size()), num_procs);

      // Distributes current tasks among processes
      task_t curr_task = task_queue.front(); // Rank 0 process gets first task
      task_queue.pop();

      for (int dest_rank = 1; dest_rank != end; ++dest_rank) {
        MPI_Send(&(task_queue.front()), 1, MPI_TASK_T, dest_rank, dest_rank, MPI_COMM_WORLD);
        task_queue.pop();
      }

      execute_task(stats, curr_task, num_new_tasks, task_buffer);

      // Process rank 0's results first
      for (int i = 0; i != num_new_tasks; ++i)
      {
        task_queue.push(task_buffer[i]);
      }

      // Receives number of new tasks from processes
      for (int src_rank = 1; src_rank != end; ++src_rank) {
        MPI_Recv(&num_new_tasks, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, &stat);
        if (num_new_tasks != 0)
        {
          MPI_Recv(task_buffer.data(), num_new_tasks, MPI_TASK_T, src_rank, 0, MPI_COMM_WORLD, &stat);
          for (int i = 0; i != num_new_tasks; ++i)
            task_queue.push(task_buffer[i]);
        }
      }
    }

    // Send termination task to processes
    task_t terminate;
    terminate.gen = -1;
    for (int i = 1; i != num_procs; ++i) {
      MPI_Send(&terminate, 1, MPI_TASK_T, i, i, MPI_COMM_WORLD);
    }

  } else {

    int num_new_tasks = 0;
    std::vector<task_t> task_buffer(Nmax);
    task_t curr_task;

    while (true) {
      MPI_Recv(&curr_task, 1, MPI_TASK_T, 0, rank, MPI_COMM_WORLD, &stat);

      // If received termination task
      if (curr_task.gen == -1) {
        break;
      }

      execute_task(stats, curr_task, num_new_tasks, task_buffer);

      MPI_Send(&num_new_tasks, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      if (num_new_tasks != 0)
        MPI_Send(task_buffer.data(), num_new_tasks, MPI_TASK_T, 0, 0, MPI_COMM_WORLD);
    }
  }
}

void run_all_tasks_master(int rank, int num_procs, metric_t &stats, params_t &params) {

  MPI_Datatype MPI_TASK_T;
  MPI_Type_create_struct(task_t_num_blocks, task_t_lengths, task_t_displs, task_t_types, &MPI_TASK_T);
  MPI_Type_commit(&MPI_TASK_T);

  MPI_Status status;
  MPI_Request request;

  if (rank == 0) {
    std::queue<task_t> task_queue;

    std::ifstream istrm(params.input_path, std::ios::binary);
    // Read initial tasks
    int count;
    istrm >> count;

    for (int i = 0; i < count; ++i) {
      task_t task;
      int type;
      istrm >> type >> task.arg_seed;
      task.type = static_cast<TaskType>(type);
      task.id = task.arg_seed;
      task.gen = 0;
      task_queue.push(task);
    }

    // Keep track of processes that are idle
    std::queue<int> free_processes;
    for (int i = 1; i != num_procs; ++i)
      free_processes.push(i);

    int num_new_tasks = 0;
    int num_tasks_executing = 0;
    std::vector<task_t> task_buffer(Nmax);

    // Send initial batch of tasks
    int end = std::min(task_queue.size(), free_processes.size());
    for (int i = 0; i != end; ++i)
    {
      int dest_rank = free_processes.front();
      int dest_tag = dest_rank;
      free_processes.pop();

      MPI_Send(&(task_queue.front()), 1, MPI_TASK_T, dest_rank, dest_tag, MPI_COMM_WORLD);
      task_queue.pop();

      ++num_tasks_executing;
    }

    while (true)
    {
      if (task_queue.empty() && num_tasks_executing == 0) break;

      MPI_Irecv(&num_new_tasks, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request);

      // While waiting for message, look for work to do
      int flag;
      while (true)
      {
        MPI_Test(&request, &flag, &status);
        if (flag) break;

        if (task_queue.empty())
        {
          MPI_Wait(&request, &status);
          break;
        }
        
        int new_tasks = 0;
        execute_task(stats, task_queue.front(), new_tasks, task_buffer);
        task_queue.pop(); // No need to update num_tasks_executing

        for (int i = 0; i != new_tasks; ++i) {
          task_queue.push(task_buffer[i]);
        }
      }

      // Receive task completion results
      int source_rank = status.MPI_SOURCE;
      free_processes.push(source_rank);
      --num_tasks_executing;
      
      if (num_new_tasks != 0)
      {
        MPI_Recv(task_buffer.data(), num_new_tasks, MPI_TASK_T, source_rank, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i != num_new_tasks; ++i)
          task_queue.push(task_buffer[i]);
      }

      // Send new tasks in reaction to task completion
      int end = std::min(task_queue.size(), free_processes.size());
      for (int i = 0; i != end; ++i)
      {
        int dest_rank = free_processes.front();
        int dest_tag = dest_rank;
        free_processes.pop();

        MPI_Send(&(task_queue.front()), 1, MPI_TASK_T, dest_rank, dest_tag, MPI_COMM_WORLD);
        task_queue.pop();

        ++num_tasks_executing;
      }
    }

    // Send termination task to processes
    task_t terminate;
    terminate.gen = -1;
    for (int i = 1; i != num_procs; ++i) {
      MPI_Send(&terminate, 1, MPI_TASK_T, i, i, MPI_COMM_WORLD);
    }

  } else {
    int num_new_tasks = 0;
    std::vector<task_t> task_buffer(Nmax);
    task_t curr_task;
  
    while (true) {
      MPI_Recv(&curr_task, 1, MPI_TASK_T, 0, rank, MPI_COMM_WORLD, &status);

      // If received termination task
      if (curr_task.gen == -1) {
        break;
      }

      execute_task(stats, curr_task, num_new_tasks, task_buffer);

      MPI_Send(&num_new_tasks, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      if (num_new_tasks != 0)
        MPI_Send(task_buffer.data(), num_new_tasks, MPI_TASK_T, 0, 0, MPI_COMM_WORLD);
    }
  }
}

void run_seq(int rank, int num_procs, metric_t& stats, params_t& params) {
  std::queue<task_t> task_queue;

  std::ifstream istrm(params.input_path, std::ios::binary);
  // Read initial tasks
  int count;
  istrm >> count;

  for (int i = 0; i < count; ++i) {
    task_t task;
    int type;
    istrm >> type >> task.arg_seed;
    task.type = static_cast<TaskType>(type);
    task.id = task.arg_seed;
    task.gen = 0;
    task_queue.push(task);
  }

  // Declare array to store generated descendant tasks
  int num_new_tasks = 0;
  std::vector<task_t> task_buffer(Nmax);
  while (!task_queue.empty()) {
    execute_task(stats, task_queue.front(), num_new_tasks, task_buffer);
    for (int i = 0; i < num_new_tasks; ++i) {
      task_queue.push(task_buffer[i]);
    }
    task_queue.pop();
  }
}