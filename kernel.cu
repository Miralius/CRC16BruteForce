#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std::string_literals;

class cudaDeviceResetException : public std::exception
{
public:
    template <
        class StringType, 
        typename = std::enable_if_t
            <!std::is_base_of
                <cudaDeviceResetException,
                std::decay_t<StringType>
            >::value
        >
    >
    cudaDeviceResetException(StringType&& exceptionDescription)
        : std::exception(std::forward<StringType>(exceptionDescription).c_str())
    {}
};

class HEXByteSymbol
{
private:
    char firstSymbol;
    char secondSymbol;

public:
    HEXByteSymbol() = default;

    HEXByteSymbol(const char firstSymbol, const char secondSymbol) noexcept
        : firstSymbol(firstSymbol)
        , secondSymbol(secondSymbol)
    {}

    operator uint8_t()
    {
        auto stringHEXByte = ""s + firstSymbol + secondSymbol;
        return static_cast<uint8_t>(std::stoul(stringHEXByte, nullptr, 16));
    }
};

std::istream& operator>>(std::istream& in, HEXByteSymbol& obj)
{
    char firstSymbol{};
    char secondSymbol{};
    in >> firstSymbol >> secondSymbol;
    if (!in)
    {
        return in;
    }
    obj = HEXByteSymbol(firstSymbol, secondSymbol);
    return in;
}

class CRC16
{
private:
    uint16_t polynome{};
    uint16_t initValue{};
    uint16_t finalXORValue{};
    bool inputReflected{};
    bool resultReflected{};
    bool initialized{};
public:
    uint16_t getPolynome() const noexcept
    {
        return polynome;
    }
    uint16_t getInitValue() const noexcept
    {
        return initValue;
    }
    uint16_t getFinalXORValue() const noexcept
    {
        return finalXORValue;
    }
    bool isInputReflected() const noexcept
    {
        return inputReflected;
    }
    bool isResultReflected() const noexcept
    {
        return resultReflected;
    }
    bool isInitialized() const noexcept
    {
        return initialized;
    }

    CRC16() = default;

    __device__ CRC16(const uint16_t polynome, const uint16_t initValue, const uint16_t finalXORValue,
        const bool inputReflected, const bool resultReflected) noexcept
        : polynome(polynome)
        , initValue(initValue)
        , finalXORValue(finalXORValue)
        , inputReflected(inputReflected)
        , resultReflected(resultReflected)
        , initialized(true)
    {}
};

std::ostream& operator<<(std::ostream& out, const CRC16& data)
{
    return out << std::noshowbase << std::hex << std::uppercase
        << "Polynome: " << data.getPolynome() << ' '
        << "Init value: " << data.getInitValue() << ' '
        << "Final XOR value: " << data.getFinalXORValue() << ' ' << std::dec
        << "Input reflected: " << (data.isInputReflected() ? "yes" : "no") << ' '
        << "Result reflected: " << (data.isResultReflected() ? "yes" : "no");
}

enum class inputResultReflected
{
    nothingReflected,
    inputReflected,
    resultReflected,
    inputAndResultReflected
};

template <typename T>
__device__ T reflect(const T& value)
{
    T reflectedByte{};
    auto bitCountOfType = sizeof(T) * 8;
    for (size_t i{}; i < bitCountOfType; i++)
    {
        uint8_t bitValue = value & (1 << i) ? 1 : 0;
        reflectedByte |= bitValue << ((bitCountOfType - 1) - i);
    }
    return reflectedByte;
}

__device__ uint16_t ComputeCRC16(const uint8_t* bytes, const size_t byteNumber, const uint16_t polynome,
    const uint16_t initValue, const uint16_t finalXorValue, const bool resultReflected)
{
    auto crc = initValue;
    for (size_t i{}; i < byteNumber; i++)
    {
        crc ^= (bytes[i] << 8);
        for (uint8_t j{}; j < 8; j++)
        {
            if ((crc & 0x8000) != 0)
            {
                crc = crc << 1 ^ polynome;
            }
            else
            {
                crc <<= 1;
            }
        }
    }
    if (resultReflected)
    {
        crc = reflect<uint16_t>(crc);
    }
    return (crc ^ finalXorValue);
}

__global__ void findCRC16Parameters(const uint8_t* data1, const uint8_t* data2, const uint8_t* data3,
    const uint8_t* data4, const uint8_t* reflectedData1, const uint8_t* reflectedData2, const uint8_t* reflectedData3,
    const uint8_t* reflectedData4, const uint16_t* crcs, const size_t size1, const size_t size2, const size_t size3,
    const size_t size4, const uint16_t finalXORValue, CRC16* result, size_t* combinationNumber)
{
    uint16_t polynome = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t initValue = blockIdx.y * blockDim.y + threadIdx.y;
    auto inputResultReflectedType = static_cast<inputResultReflected>(blockIdx.z * blockDim.z + threadIdx.z);
    bool inputReflected{};
    auto resultReflected = inputResultReflectedType == inputResultReflected::resultReflected
        || inputResultReflectedType == inputResultReflected::inputAndResultReflected;
    if (inputResultReflectedType == inputResultReflected::inputReflected
        || inputResultReflectedType == inputResultReflected::inputAndResultReflected)
    {
        inputReflected = true;
    }
    if (ComputeCRC16(inputReflected ? reflectedData1 : data1, size1, polynome, initValue, finalXORValue,
        resultReflected) == crcs[0]
        && ComputeCRC16(inputReflected ? reflectedData2 : data2, size2, polynome, initValue, finalXORValue,
            resultReflected) == crcs[1]
        && ComputeCRC16(inputReflected ? reflectedData3 : data3, size3, polynome, initValue, finalXORValue,
            resultReflected) == crcs[2]
        && ComputeCRC16(inputReflected ? reflectedData4 : data4, size4, polynome, initValue, finalXORValue,
            resultReflected) == crcs[3])
    {
        atomicAdd(combinationNumber, 1);
        *result = CRC16(polynome, initValue, finalXORValue, inputReflected, resultReflected);
    }
}

template <class T, class StringType>
cudaError_t cudaMallocAndMemcpyData(T*& pointer, const T& data, StringType&& string)
{
    cudaError_t cudaStatus{};
    cudaStatus = cudaMalloc(&pointer, sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaMalloc for "s + string + " failed!\n" + cudaGetErrorString(cudaStatus));
    }
    return cudaMemcpy(pointer, &data, sizeof(T), cudaMemcpyHostToDevice);
}

template <class T, class StringType>
cudaError_t cudaMallocAndMemcpyData(T*& pointer, const std::vector<T>& data, StringType&& string)
{
    cudaError_t cudaStatus{};
    cudaStatus = cudaMalloc(&pointer, data.size() * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaMalloc for "s + string + " failed!\n" + cudaGetErrorString(cudaStatus));
    }
    return cudaMemcpy(pointer, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <class T, class StringType>
void cudaMallocAndMemcpyData(T**& pointers, const std::vector<std::vector<T>>& vectors, StringType&& string)
{
    for (uint8_t currentPointerNumber{}; currentPointerNumber < vectors.size(); currentPointerNumber++)
    {
        auto cudaStatus = cudaMallocAndMemcpyData(pointers[currentPointerNumber], vectors.at(currentPointerNumber),
            "data"s + std::to_string(currentPointerNumber));
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for "s + std::forward<StringType>(string)
                + std::to_string(currentPointerNumber) + " failed!\n" + cudaGetErrorString(cudaStatus));
        }
    }
}

template <typename T, typename StringType>
T loadingValueFromFileInHEX(StringType&& nameFile)
{
    std::ifstream in(nameFile);
    T value{};
    if (!in.fail())
    {
        in >> std::hex >> std::uppercase >> value;
        in.close();
    }
    else
    {
        in.close();
        throw std::runtime_error(std::forward<StringType>(nameFile) + " is not found!");
    }

    return value;
}

template <typename T, typename StringType>
std::vector<T> loadingFileInHEX(StringType&& nameFile)
{
    std::ifstream in(nameFile);
    auto name = "File "s + std::forward<StringType>(nameFile);
    std::vector<T> vector;
    if (!in.fail())
    {
        T buffer{};
        while (in >> std::hex >> std::uppercase >> buffer)
        {
            vector.emplace_back(buffer);
        }
        in.close();
    }
    else
    {
        in.close();
        throw std::runtime_error(name + " is not found!");
    }
    if (vector.size() == 0)
    {
        throw std::runtime_error(name + " is empty or contains wrong data!");
    }

    return vector;
}

template <typename StringType>
std::vector<uint8_t> loadingBytesInHEXFromFile(StringType&& nameFile)
{
    std::ifstream in(nameFile);
    auto name = "File "s + std::forward<StringType>(nameFile);
    std::vector<uint8_t> byteVector{};
    if (!in.fail())
    {
        HEXByteSymbol byte{};
        while (in >> byte)
        {
            byteVector.emplace_back(static_cast<uint8_t>(byte));
        }
        in.close();
    }
    else
    {
        in.close();
        throw std::runtime_error(name + " is not found!");
    }
    if (byteVector.size() == 0)
    {
        throw std::runtime_error(name + " is empty or contains wrong data!");
    }
    return byteVector;
}

template <typename StringType>
void addEntryIntoFile(CRC16&& data, StringType&& nameFile)
{
    std::ofstream output(nameFile, std::ios_base::app);
    if (!output)
    {
        output.close();
        throw std::runtime_error("Writing into file " + std::forward<StringType>(nameFile) + " is impossible!");
    }
    output << std::move(data) << '\n';
    output.close();
}

template <class T, typename StringType>
void eraseFileAndWriteValueInHEX(T&& data, StringType&& nameFile)
{
    std::ofstream output(nameFile);
    if (!output)
    {
        output.close();
        throw std::runtime_error("Writing into file " + std::forward<StringType>(nameFile) + " is impossible!");
    }
    output << std::noshowbase << std::hex << std::uppercase << std::move(data);
    output.close();
}

CRC16 bruteForceCRC16WithGPU(const uint16_t finalXORValue, const std::vector<std::vector<uint8_t>>& data,
    const std::vector<std::vector<uint8_t>>& reflectedData, const std::vector<uint16_t>& crcs)
{
    cudaError_t cudaStatus{};
    uint16_t* crcsPointer{};
    CRC16* result{};
    size_t* combinationNumberPtr{};
    auto sizes = new size_t[4];
    auto dataPointers = new uint8_t*[4];
    auto reflectedDataPointers = new uint8_t*[4];
    try
    {
        cudaStatus = cudaSetDevice(0);
        if (cudaSetDevice(0) != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n"s
                + cudaGetErrorString(cudaStatus));
        }

        for (uint8_t currentSizeNumber{}; currentSizeNumber < data.size(); currentSizeNumber++)
        {
            sizes[currentSizeNumber] = data.at(currentSizeNumber).size();
        }

        cudaMallocAndMemcpyData(dataPointers, data, "data"s);
        cudaMallocAndMemcpyData(reflectedDataPointers, reflectedData, "reflected data"s);

        cudaStatus = cudaMallocAndMemcpyData(crcsPointer, crcs, "crcs"s);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for crcs failed!\n"s + cudaGetErrorString(cudaStatus));
        }

        CRC16 computedResult{};
        cudaStatus = cudaMallocAndMemcpyData(result, computedResult, "result"s);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for result failed!\n"s + cudaGetErrorString(cudaStatus));
        }

        size_t combinationNumber{};
        cudaStatus = cudaMallocAndMemcpyData(combinationNumberPtr, combinationNumber, "combination number"s);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for combination number failed!\n"s + cudaGetErrorString(cudaStatus));
        }

        uint16_t polynomeCount{ UINT16_MAX };
        uint16_t initValueCount{ UINT16_MAX };
        uint8_t inputOrResultReflectedCombinationCount{ 4 };
        dim3 threadPerBlock(32, 32, 1);
        dim3 blocks(polynomeCount / threadPerBlock.x, initValueCount / threadPerBlock.y,
            inputOrResultReflectedCombinationCount / threadPerBlock.z);

        findCRC16Parameters<<<blocks, threadPerBlock>>>(dataPointers[0], dataPointers[1], dataPointers[2],
            dataPointers[3], reflectedDataPointers[0], reflectedDataPointers[1], reflectedDataPointers[2],
            reflectedDataPointers[3], crcsPointer, sizes[0], sizes[1], sizes[2], sizes[3], finalXORValue, result,
            combinationNumberPtr);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("Kernel computing failed!\n"s + cudaGetErrorString(cudaStatus));
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaDeviceSynchronize failed!\n"s + cudaGetErrorString(cudaStatus));
        }

        cudaStatus = cudaMemcpy(&computedResult, result, sizeof(CRC16), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for computed result failed!\n"s + cudaGetErrorString(cudaStatus));
        }

        cudaStatus = cudaMemcpy(&combinationNumber, combinationNumberPtr, sizeof(size_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for combination number failed!\n"s + cudaGetErrorString(cudaStatus));
        }

        if (combinationNumber > 1)
        {
            throw std::runtime_error("Two or more combination were found!\n");
        }

        cudaFree(result);
        cudaFree(crcsPointer);
        cudaFree(combinationNumberPtr);
        for (uint8_t currentPointerNumber{}; currentPointerNumber < data.size(); currentPointerNumber++)
        {
            cudaFree(dataPointers[currentPointerNumber]);
            cudaFree(reflectedDataPointers[currentPointerNumber]);
        }
        delete[] sizes;
        delete[] dataPointers;
        delete[] reflectedDataPointers;

        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            throw cudaDeviceResetException("cudaDeviceReset failed!\n"s + cudaGetErrorString(cudaStatus));
        }

        return computedResult;
    }
    catch (std::runtime_error ex)
    {
        cudaFree(result);
        cudaFree(crcsPointer);
        cudaFree(combinationNumberPtr);
        for (uint8_t currentPointerNumber{}; currentPointerNumber < data.size(); currentPointerNumber++)
        {
            cudaFree(dataPointers[currentPointerNumber]);
            cudaFree(reflectedDataPointers[currentPointerNumber]);
        }
        delete[] sizes;
        delete[] dataPointers;
        delete[] reflectedDataPointers;

        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaDeviceReset failed! " << cudaGetErrorString(cudaStatus) << '\n';
        }
        throw;
    }
}

bool XNOR(const bool firstCondition, const bool secondCondition)
{
    return firstCondition == secondCondition;
}

template <typename T>
std::vector<T> reflect(const std::vector<T>& values)
{
    std::vector<T> reflected;
    auto bitCountOfType = sizeof(T) * 8;
    for (const auto& value : values)
    {
        T reflectedValue{};
        for (size_t i{}; i < bitCountOfType; i++)
        {
            uint8_t bitValue = value & (1 << i) ? 1 : 0;
            reflectedValue |= bitValue << ((bitCountOfType - 1) - i);
        }
        reflected.emplace_back(reflectedValue);
    }
    return reflected;
}

template <class T>
void showProgress(const T completed, const T total)
{
    auto percent = std::trunc(10000 * (static_cast<float>(completed) / total)) / 100;
    std::cout << '\r' << "Completed: " << std::dec << std::setw(6) << percent << '%';
}

template <class TimePointType, class IntType>
void showRemainingExecutionTime(TimePointType&& start, TimePointType&& end, const IntType remainingOperationNumber)
{
    using namespace std::chrono;
    using namespace std::chrono_literals;
    auto durationInSeconds = duration_cast<seconds>(end - start) * remainingOperationNumber;
    if (durationInSeconds != 0s)
    {
        std::cout << "Remaining time: " << std::dec;
        auto durationInHours = duration_cast<hours>(durationInSeconds);
        if (durationInHours != 0h)
        {
            std::cout << std::setw(6) << durationInHours.count() << "h, ";
            durationInSeconds -= duration_cast<seconds>(durationInHours);
        }
        auto durationInMinutes = duration_cast<minutes>(durationInSeconds);
        if (durationInMinutes != 0min || durationInHours != 0h)
        {
            std::cout << std::setw(2) << durationInMinutes.count() << "min, ";
            durationInSeconds -= duration_cast<seconds>(durationInMinutes);
        }
        if (durationInSeconds != 0s || durationInMinutes != 0min || durationInHours != 0h)
        {
            std::cout << std::setw(2) << durationInSeconds.count() << "s";
        }
    }
}

void calculateCRC16WithGPU(std::vector<std::vector<uint8_t>>&& data, std::vector<uint16_t>&& crcs)
{
    std::vector<std::vector<uint8_t>> reflectedData
    {
        reflect(data.at(0)),
        reflect(data.at(1)),
        reflect(data.at(2)),
        reflect(data.at(3))
    };
    bool overflowed{};
    uint16_t finalXORValue{ UINT16_MAX };
    auto progressFileName = "Progress.txt"s;
    try {
        finalXORValue = loadingValueFromFileInHEX<uint16_t>(progressFileName);
        overflowed = true;
    }
    catch (std::runtime_error ex)
    {}
    std::chrono::steady_clock::time_point start{};
    std::chrono::steady_clock::time_point end{};
    for (; XNOR(finalXORValue < UINT16_MAX, overflowed); finalXORValue++)
    {
        uint16_t completed = overflowed ? finalXORValue : 0;
        showProgress(completed, UINT16_MAX);
        std::cout << " Final XOR value: " << std::noshowbase << std::hex << std::uppercase << finalXORValue << ' ';
        showRemainingExecutionTime(std::move(start), std::move(end), static_cast<uint16_t>(UINT16_MAX - completed));
        start = std::chrono::steady_clock::now();
        eraseFileAndWriteValueInHEX(finalXORValue, progressFileName);
        auto result = bruteForceCRC16WithGPU(finalXORValue, data, reflectedData, crcs);
        if (result.isInitialized())
        {
            std::cout << '\n' << result << '\n';
            addEntryIntoFile(std::move(result), "Results.txt"s);
        }
        if (finalXORValue == UINT16_MAX)
        {
            overflowed = true;
        }
        end = std::chrono::steady_clock::now();
    }
}

void initAndStartCalculating()
{
    std::vector<std::vector<uint8_t>> data
    {
        loadingBytesInHEXFromFile("1.txt"s),
        loadingBytesInHEXFromFile("2.txt"s),
        loadingBytesInHEXFromFile("3.txt"s),
        loadingBytesInHEXFromFile("4.txt"s)
    };
    auto crcs = loadingFileInHEX<uint16_t>("CRC.txt"s);
    calculateCRC16WithGPU(std::move(data), std::move(crcs));
}

int main()
{
    try
    {
        std::cout << "CRC16 brute force with GPU...\n";
        initAndStartCalculating();
    }
    catch (std::runtime_error ex)
    {
        std::cerr << "Error! " << ex.what() << '\n';
        return -1;
    }
    catch (cudaDeviceResetException ex)
    {
        std::cerr << "CUDA device reset error! " << ex.what() << '\n';
        return 1;
    }
    catch (std::exception ex)
    {
        std::cerr << "Unknown error! " << ex.what() << '\n';
        return -1;
    }
    return 0;
}