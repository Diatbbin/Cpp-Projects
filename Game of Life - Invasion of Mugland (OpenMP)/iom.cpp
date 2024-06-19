#include "iom.h"
#include "omp.h"
#include <iostream>

int shouldReproduce(int x, int y, const std::vector<std::vector<int>>& world);
int countFriendlyNeighbours(int x, int y, const std::vector<std::vector<int>>& world);
bool hasHostileNeighbors(const int x, const int y, const std::vector<std::vector<int>>& world);

int iom(int nThreads, int nGenerations, std::vector<std::vector<int>>& startWorld, int nRows, int nCols, int nInvasions, std::vector<int> invasionTimes, std::vector<std::vector<std::vector<int>>> invasionPlans) {
    
    std::vector<std::vector<int>> mainWorld = startWorld;
    int invasionPtr = 0;
    int deathToll = 0;
    bool hasInvasion = false;
	
    omp_set_num_threads(nThreads);

    for (int gen = 1; gen != nGenerations + 1; ++gen)
    {
        const std::vector<std::vector<int>> dupWorld = mainWorld;
        
        if (invasionPtr < nInvasions && invasionTimes[invasionPtr] == gen) hasInvasion = true;
        
	#pragma omp parallel for reduction(+:deathToll) collapse(2)
        for (int x = 0; x != nRows; ++x)
        for (int y = 0; y != nCols; ++y)
        {
            if (dupWorld[x][y] == 0)
            {
                mainWorld[x][y] = shouldReproduce(x, y, dupWorld);
            }
            else if (hasHostileNeighbors(x, y, dupWorld))
            {
                mainWorld[x][y] = 0;
                ++deathToll;
            }
            else
            {
                int count = countFriendlyNeighbours(x, y, dupWorld);
                if (count < 2 || count >= 4)
                {
                    mainWorld[x][y] = 0;
                }
                else
                {
                    mainWorld[x][y] = dupWorld[x][y];
                }
            }

            if (hasInvasion && invasionPlans[invasionPtr][x][y])
            {
                int invaderFaction = invasionPlans[invasionPtr][x][y];
                mainWorld[x][y] = invaderFaction;

                if (dupWorld[x][y] != 0) 
                {
                    ++deathToll;
                }    
            }
        }
        invasionPtr = (hasInvasion) ? invasionPtr + 1 : invasionPtr;
        hasInvasion = false;
    }

    return deathToll;
}

int shouldReproduce(int x, int y, const std::vector<std::vector<int>>& world)
{
    std::vector<int> count(9, 0);

    int moves[8][2] = {
        {-1,-1},
        {-1,0},
        {-1,1},
        {0,-1},
        {0,1},
        {1,-1},
        {1,0},
        {1,1}
    };

    for (int i = 0; i < 8; i++) {
            int newRow = (moves[i][0] + x + world.size()) % world.size(); 
            int newCol = (moves[i][1] + y + world[0].size()) % world[0].size();

            if (world[newRow][newCol])
            {
                ++count[world[newRow][newCol] - 1];
            }
    }
    
    for (int i = 8; i >= 0; --i)
    {
        if (count[i] == 3) return (i + 1);
    }

    return 0;
}

int countFriendlyNeighbours(int x, int y, const std::vector<std::vector<int>>& world) {
    int count = 0;

    int moves[8][2] = {
        {-1,-1},
        {-1,0},
        {-1,1},
        {0,-1},
        {0,1},
        {1,-1},
        {1,0},
        {1,1}
    };

    for (int i = 0; i < 8; i++) {
            int newRow = (moves[i][0] + x + world.size()) % world.size(); 
            int newCol = (moves[i][1] + y + world[0].size()) % world[0].size();

            if (world[x][y] == world[newRow][newCol]) {
                    ++count;
            }
    
    }
    return count;
}


bool hasHostileNeighbors(const int x, const int y, const std::vector<std::vector<int>>& world)
{
    int faction = world[x][y];

    int moves[8][2] = {
        {-1,-1},
        {-1,0},
        {-1,1},
        {0,-1},
        {0,1},
        {1,-1},
        {1,0},
        {1,1}
    };

    for (int i = 0; i != 8; ++i)
    {
        int newRow = (moves[i][0] + x + world.size()) % world.size(); 
        int newCol = (moves[i][1] + y + world[0].size()) % world[0].size();

        if (world[newRow][newCol] && world[newRow][newCol] != faction) return true;
    }

    return false;
}
