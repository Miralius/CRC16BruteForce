#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CRC16.h"
#include "CudaDeviceResetException.h"

using namespace std::string_literals;

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
        crc ^= (static_cast<uint16_t>(bytes[i]) << 8);
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
        throw std::runtime_error("cudaMalloc for "s + string + " failed: " + cudaGetErrorString(cudaStatus));
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
        throw std::runtime_error("cudaMalloc for "s + string + " failed: " + cudaGetErrorString(cudaStatus));
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
                + std::to_string(currentPointerNumber) + " failed: " + cudaGetErrorString(cudaStatus));
        }
    }
}

CRC16 bruteForceCRC16WithGPU(const uint16_t finalXORValue, const std::vector<std::vector<uint8_t>>& data,
    const std::vector<std::vector<uint8_t>>& reflectedData, const std::vector<uint16_t>& crcs)
{
    cudaError_t cudaStatus{};
    uint16_t* crcsPointer{};
    CRC16* result{};
    size_t* combinationNumberPtr{};
    auto sizes = new size_t[4];
    auto dataPointers = new uint8_t * [4];
    auto reflectedDataPointers = new uint8_t * [4];
    try
    {
        cudaStatus = cudaSetDevice(0);
        if (cudaSetDevice(0) != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed: Do you have a CUDA-capable GPU installed? "s
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
            throw std::runtime_error("cudaMemcpy for crcs failed: "s + cudaGetErrorString(cudaStatus));
        }

        CRC16 computedResult{};
        cudaStatus = cudaMallocAndMemcpyData(result, computedResult, "result"s);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for result failed: "s + cudaGetErrorString(cudaStatus));
        }

        size_t combinationNumber{};
        cudaStatus = cudaMallocAndMemcpyData(combinationNumberPtr, combinationNumber, "combination number"s);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for combination number failed: "s + cudaGetErrorString(cudaStatus));
        }

        uint16_t polynomeCount{ UINT16_MAX };
        uint16_t initValueCount{ UINT16_MAX };
        uint8_t inputOrResultReflectedCombinationCount{ 4 };
        dim3 threadPerBlock(32, 32, 1);
        dim3 blocks(polynomeCount / threadPerBlock.x, initValueCount / threadPerBlock.y,
            inputOrResultReflectedCombinationCount / threadPerBlock.z);

        findCRC16Parameters << <blocks, threadPerBlock >> > (dataPointers[0], dataPointers[1], dataPointers[2],
            dataPointers[3], reflectedDataPointers[0], reflectedDataPointers[1], reflectedDataPointers[2],
            reflectedDataPointers[3], crcsPointer, sizes[0], sizes[1], sizes[2], sizes[3], finalXORValue, result,
            combinationNumberPtr);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("Kernel computing failed: "s + cudaGetErrorString(cudaStatus));
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaDeviceSynchronize failed: "s + cudaGetErrorString(cudaStatus));
        }

        cudaStatus = cudaMemcpy(&computedResult, result, sizeof(CRC16), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for computed result failed: "s + cudaGetErrorString(cudaStatus));
        }

        cudaStatus = cudaMemcpy(&combinationNumber, combinationNumberPtr, sizeof(size_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::runtime_error("cudaMemcpy for combination number failed: "s + cudaGetErrorString(cudaStatus));
        }

        if (combinationNumber > 1)
        {
            throw std::runtime_error("Two or more combination were found! ");
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
            throw CudaDeviceResetException("cudaDeviceReset failed: "s + cudaGetErrorString(cudaStatus));
        }

        return computedResult;
    }
    catch (std::runtime_error ex)
    {
        auto errorString = static_cast<std::string>(ex.what()) + ", " + cudaGetErrorString(cudaStatus);
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
            std::cerr << '\n' << "cudaDeviceReset failed: " << cudaGetErrorString(cudaStatus);
        }
        throw;
    }
}