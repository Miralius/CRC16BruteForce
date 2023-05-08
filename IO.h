#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "HEXByteSymbol.h"

using namespace std::string_literals;

enum class LogLevel
{
    FATAL,
    INFO
};

inline decltype(auto) logLevelToString(const LogLevel logLevel)
{
    switch (logLevel)
    {
    case LogLevel::FATAL:
        return "FATAL"s;
    case LogLevel::INFO:
        return "INFO"s;
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

template <class T, typename StringType>
void addEntryIntoFile(T&& data, StringType&& nameFile)
{
    std::ofstream output(nameFile, std::ios_base::app);
    if (!output)
    {
        output.close();
        throw std::runtime_error("Writing into file "s + std::forward<StringType>(nameFile) + " is impossible!");
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

template <typename ErrorStringType>
void writeLogEntry(ErrorStringType&& description, const LogLevel logLevel)
{
    const auto timePoint = std::chrono::zoned_time{ std::chrono::current_zone(), std::chrono::system_clock::now() };
    const auto logString = std::format("{} [{}]: {}", timePoint, logLevelToString(logLevel), std::move(description));
    addEntryIntoFile(std::move(logString), "logFile.txt"s);
}