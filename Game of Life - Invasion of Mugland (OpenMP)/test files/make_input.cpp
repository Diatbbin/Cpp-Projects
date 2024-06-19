#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

int main()
{
    srand(std::time(nullptr));

    int nRows = 50;
    int nCols = 60;
    int nGenerations = 1000000;

    std::string fileName = "sample_input.in";
    std::ofstream outputFile(fileName);

    // Generations
    outputFile << nGenerations << "\n";
    
    // Dimensions
    outputFile << nRows << "\n";
    outputFile << nCols << "\n";

    // Start World
    int vHalf = nRows / 2;
    int hHalf = nCols / 2;

    for (int i = 0; i != nRows; ++i)
    {
        for (int j = 0; j != nCols; ++j)
        {
            if (j != 0) outputFile << " ";

            float r = ((float) rand() / (RAND_MAX));
            if (r < 0.6)
            {
                outputFile << 0;
            }
            else
            {

                int faction; 
                if (i <= vHalf && j <= hHalf) faction = 1;
                else if (i <= vHalf && j > hHalf) faction = 2;
                else if (i > vHalf && j > hHalf) faction = 4;
                else faction = 3;

                outputFile << faction;
            }
        }
        outputFile << "\n";
    }

    // Invasions
    outputFile << 1000 << "\n";

    for (int i = 0; i != 1000; ++i)
    {
        int idx = (rand() % 1000) + 1;
        int gen = i * 1000 + idx;
        outputFile << gen << "\n";
        for (int i = 0; i != nRows; ++i)
        {
            for (int j = 0; j != nCols; ++j)
            {
                if (j != 0) outputFile << " ";

                float invade = ((float) rand() / (RAND_MAX));

                if (invade > 0.1)
                {
                    outputFile << 0;
                    continue;
                }

                int faction = (rand() % 9) + 1;
                outputFile << faction;
            }
            outputFile << "\n";
        }
    }
}