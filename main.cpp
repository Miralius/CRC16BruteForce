#include "main.h"

int main()
{
    try
    {
        std::cout << "CRC16 brute force with GPU...\n";
        initAndStartCalculating();
    }
    catch (std::runtime_error ex)
    {
        writeLogEntry(ex.what(), LogLevel::FATAL);
        std::cerr << '\n' << "Error! " << ex.what();
        return -1;
    }
    catch (CudaDeviceResetException ex)
    {
        writeLogEntry(ex.what(), LogLevel::FATAL);
        std::cerr << '\n' << "CUDA device reset error! " << ex.what();
        return 1;
    }
    catch (std::exception ex)
    {
        writeLogEntry(ex.what(), LogLevel::FATAL);
        std::cerr << '\n' << "Unknown error! " << ex.what();
        return -1;
    }
    return 0;
}