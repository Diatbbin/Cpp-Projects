#include <iostream>
#include <fstream>
#include <vector>
#include "iom.h"

int read_param(std::ifstream& input_file, std::string& line, int& param);
int read_world_layout(std::ifstream& input_file, std::string& line, std::vector<std::vector<int>>& world, int n_rows, int n_cols);

int main(int argc, char *argv[])
{   
    int n_threads{};
    int n_generations{};
    int n_rows{};
    int n_cols{};
    int n_invasions{};

    std::ifstream input_file;
    std::ofstream output_file;

    std::string curr_line{};

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <INPUT_PATH> <OUTPUT_PATH> <NUM_THREADS>" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "<INPUT_PATH>: " << argv[1] << std::endl;
    std::cout << "<OUTPUT_PATH>: " << argv[2] << std::endl;
    std::cout << "<NUM_THREADS>: " << argv[3] << std::endl;

    
    input_file.open(argv[1]);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open " << argv[1] << " for reading. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    
    output_file.open(argv[2]);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open " << argv[2] << " for writing. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    try
    {
        n_threads = std::stoi(argv[3]);
    }
    catch (std::invalid_argument const& ex)
    {
        std::cerr << "Invalid argument for <NUM_THREADS>: " << argv[3] << ". Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (std::out_of_range const& ex)
    {
        std::cerr << "<NUM_THREADS> out of int range: " << argv[3] << ". Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (n_threads < 1) {
        std::cerr << "<NUM_THREADS> has invalid value: " << argv[3] << ". Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read n_generations
    if (read_param(input_file, curr_line, n_generations) == -1)
    {
        std::cerr << "Failed to read N_GENERATIONS. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }  

    // Read n_rows
    if (read_param(input_file, curr_line, n_rows) == -1)
    {
        std::cerr << "Failed to read N_ROWS. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read n_cols
    if (read_param(input_file, curr_line, n_cols) == -1)
    {
        std::cerr << "Failed to read N_COLS. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (n_rows == 0 || n_cols == 0)
    {
        std::cerr << "N_ROWS or N_COLS is 0. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<int>> start_world(n_rows, std::vector<int>(n_cols));
    if (read_world_layout(input_file, curr_line, start_world, n_rows, n_cols) == -1)
    {
        std::cerr << "Failed to read STARTING_WORLD. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (read_param(input_file, curr_line, n_invasions) == -1) {
        std::cerr << "Failed to read N_INVASIONS. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read invasions
    std::vector<int> invasion_times(n_invasions);
    std::vector<std::vector<std::vector<int>>> invasion_plans(
        n_invasions, 
        std::vector<std::vector<int>>(n_rows, std::vector<int>(n_cols)));
    for (int i = 0; i < n_invasions; i++)
    {
        if (read_param(input_file, curr_line, invasion_times[i]))
        {
            std::cerr << "Failed to read INVASION_TIME. Aborting..." << std::endl; 
            exit(EXIT_FAILURE);
        }

        if (read_world_layout(input_file, curr_line, invasion_plans[i], n_rows, n_cols))
        {
            std::cerr << "Failed to read INVASION_PLANS. Aborting..." << std::endl; 
            exit(EXIT_FAILURE);
        }
    }

    int death_toll = iom(n_threads, n_generations, start_world, n_rows, n_cols, n_invasions, invasion_times, invasion_plans);

    output_file << death_toll;
}

int read_world_layout(std::ifstream& input_file, std::string& line, std::vector<std::vector<int>>& world, int n_rows, int n_cols) {
    for (int row = 0; row < n_rows; row++) {
        
        std::getline(input_file, line);
        if (input_file.fail())
        {
            return -1;
        }

        for (int col = 0; col < n_cols; col++)
        {
            size_t pos{};
            try 
            {
                int cell = std::stoi(line, &pos, 10);
                world.at(row).at(col) = cell;
            }
            catch (const std::invalid_argument& e)
            {
                return -1;
            }
            catch (const std::out_of_range& e)
            {
                return -1;
            }
            line = line.substr(pos);
        }
    }
    return 0;
}

int read_param(std::ifstream& input_file, std::string& line, int& param)
{
    std::getline(input_file, line);
    
    if (input_file.fail())
    {
        return -1;
    }

    try 
    {
        param = std::stoi(line);
    }
    catch (const std::exception& e)
    {
        return -1;
    }
    return 0;
    
}
