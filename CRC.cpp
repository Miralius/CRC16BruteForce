#include<iostream>
#include<fstream>
#include<sstream>
#include<iomanip>
#include<amp.h>

using namespace std;
using namespace concurrency;
using std::vector;

inline void error(const string& s)
{
	throw runtime_error(s);
}

class crc16
{
private:
	unsigned int polynome, initVal, finalXorVal, inputReflected, resultReflected;

public:
	unsigned int p() const restrict(amp, cpu) { return polynome; }
	unsigned int iV() const restrict(amp, cpu) { return initVal; }
	unsigned int fXV() const restrict(amp, cpu) { return finalXorVal; }
	unsigned int iR() const restrict(amp, cpu) { return inputReflected; }
	unsigned int rR() const restrict(amp, cpu) { return resultReflected; }

	crc16() restrict(amp, cpu)
	{
		polynome = 0x0;
		initVal = 0x0;
		finalXorVal = 0x0;
		inputReflected = 0;
		resultReflected = 0;
	}

	crc16(unsigned int setPolynome, unsigned int setInitVal, unsigned int setFinalXorVal, unsigned int setInputReflected, unsigned int setResultReflected) restrict(amp, cpu) :
		polynome(setPolynome), initVal(setInitVal), finalXorVal(setFinalXorVal), inputReflected(setInputReflected), resultReflected(setResultReflected)
	{}

	crc16(const crc16& obj) restrict(amp, cpu)
	{
		polynome = obj.polynome;
		initVal = obj.initVal;
		finalXorVal = obj.finalXorVal;
		inputReflected = obj.inputReflected;
		resultReflected = obj.resultReflected;
	}

	crc16& operator=(const crc16& obj) restrict(amp, cpu)
	{
		if (this == &obj) return *this;
		polynome = obj.polynome;
		initVal = obj.initVal;
		finalXorVal = obj.finalXorVal;
		inputReflected = obj.inputReflected;
		resultReflected = obj.resultReflected;
		return *this;
	}
};

ostream& operator<<(ostream& output, const crc16& date)
{
	return output << noshowbase << hex << uppercase << date.p() << ' ' << date.iV() << ' ' << date.fXV() << ' ' << dec << date.iR() << ' ' << date.rR() << endl;
}

istream& operator>>(istream& input, crc16& date)
{
	unsigned int polynome, initVal, finalXorVal, inputReflected, resultReflected;
	input >> hex >> polynome >> initVal >> finalXorVal >> inputReflected >> resultReflected;
	if (!input) return input;
	date = crc16(polynome, initVal, finalXorVal, inputReflected, resultReflected);
	return input;
}

bool operator== (const crc16& obj1, const crc16& obj2) restrict(amp, cpu)
{
	return (obj1.p() == obj2.p()) && (obj1.iV() == obj2.iV()) && (obj1.fXV() == obj2.fXV()) && (obj1.iR() == obj2.iR()) && (obj1.rR() == obj2.rR());
}

bool operator!= (const crc16& obj1, const crc16& obj2) restrict(amp, cpu) {
	return !(obj1 == obj2);
}

template <typename T> vector<T> loadingFile(string nameFile)
{
	ifstream in(nameFile);
	vector<T> vectorName;
	if (!in.fail())
	{
		T buffer;
		while (in >> hex >> uppercase >> buffer)
		{
			if (in.eof()) break;
			vectorName.push_back(buffer);
		}
		in.close();
		if (vectorName.size() == 0) error("Файл " + nameFile + " пуст или содержит неверные данные!");
	}
	else if (nameFile != "Progress.txt") error("Файл " + nameFile + " не найден!");
	return vectorName;
}

template <typename T> void writingFile(vector<T> date, string nameFile)
{
	ofstream output(nameFile, ios_base::app);
	if (!output) error("Запись в файл " + nameFile + " невозможна!");
	for (int i = 0; i < date.size(); i++) output << date[i];
}

template <typename T> void writingFile(T date, string nameFile)
{
	ofstream output(nameFile, ios_base::app);
	if (!output) error("Запись в файл " + nameFile + " невозможна!");
	output << hex << noshowbase << uppercase << date << endl;
}

void eraseFile(string nameFile)
{
	ofstream output(nameFile);
	if (!output) error("Невозможно стереть файл " + nameFile + "!");
}

template <typename T> T reflect(T val) restrict(amp, cpu)
{
	T resByte = 0;
	for (int i = 0; i < 16; i++) if ((val & (1 << i)) != 0) resByte |= (1 << (15 - i));
	return resByte;
}

template <typename T> vector<T> reflect(vector<T> val)
{
	vector<T> reflected;
	for (int j = 0; j < val.size(); j++)
	{
		T resByte = 0;
		for (int i = 0; i < 8; i++) if ((val.at(j) & (1 << i)) != 0) resByte |= (1 << (7 - i));
		reflected.push_back(resByte);
	}
	return reflected;
}

unsigned int Compute_CRC16_Simple(array_view<unsigned int> bytes, unsigned int polynome, unsigned int initVal, unsigned int finalXorVal, int resultReflected) restrict(amp, cpu)
{
	unsigned int crc = initVal;
	for (int i = 0; i < bytes.extent.size(); i++)
	{
		crc ^= (bytes[i] << 8);
		for (int j = 0; j < 8; j++)
		{
			if ((crc & 0x8000) != 0)
			{
				crc = ((crc << 1) ^ polynome);
			}
			else
			{
				crc <<= 1;
			}
		}
	}
	crc &= 0xFFFF;
	if (resultReflected) crc = reflect<unsigned int>(crc);
	return (crc ^ finalXorVal);
}

unsigned int Compute_CRC16_Simple(vector<unsigned int> bytes, unsigned int polynome, unsigned int initVal, unsigned int finalXorVal, int resultReflected)
{
	unsigned int crc = initVal;
	for (int i = 0; i < bytes.size(); i++)
	{
		crc ^= (bytes.at(i) << 8);
		for (int j = 0; j < 8; j++)
		{
			if ((crc & 0x8000) != 0)
			{
				crc = ((crc << 1) ^ polynome);
			}
			else
			{
				crc <<= 1;
			}
		}
	}
	crc &= 0xFFFF;
	if (resultReflected) crc = reflect<unsigned int>(crc);
	return (crc ^ finalXorVal);
}

atomic<int> countResults(0), sizeOfParts(0);

void processing(int NOW, int MAX)
{
	float proc, nowf, maxf;
	if (NOW == MAX) proc = 100.;
	else
	{
		nowf = NOW;
		maxf = MAX;
		proc = trunc(10000 * (nowf / maxf)) / 100;
	}
	cout << '\r' << "Выполнено: " << dec << setw(6) << proc << "% Найдено комбинаций: " << countResults;
}

void calculateCRCwithGPU(string nameInputFile1, string nameInputFile2, string nameInputFile3, string nameInputFile4, string nameOutputFile, unsigned int resultCRC1, unsigned int resultCRC2, unsigned int resultCRC3, unsigned int resultCRC4, unsigned int resultCRC1SVOP, unsigned int resultCRC2SVOP, unsigned int resultCRC3SVOP, unsigned int resultCRC4SVOP, unsigned int finalXorVal, int inputReflected, int resultReflected)
{
	vector<unsigned int> bytes1 = loadingFile<unsigned int>(nameInputFile1);
	vector<unsigned int> bytes2 = loadingFile<unsigned int>(nameInputFile2);
	vector<unsigned int> bytes3 = loadingFile<unsigned int>(nameInputFile3);
	vector<unsigned int> bytes4 = loadingFile<unsigned int>(nameInputFile4);
	if (inputReflected)
	{
		bytes1 = reflect<unsigned int>(bytes1);
		bytes2 = reflect<unsigned int>(bytes2);
		bytes3 = reflect<unsigned int>(bytes3);
		bytes4 = reflect<unsigned int>(bytes4);
	}
	int sizeOfLambdaFunction = sizeOfParts;
	vector<crc16> results;
	crc16 crc;
	for (int i = 0; i < 65536; i++) results.push_back(crc);
	Concurrency::extent<2> e(0x10000, sizeOfLambdaFunction);
	array_view<unsigned int> b1(bytes1.size(), bytes1);
	array_view<unsigned int> b2(bytes2.size(), bytes2);
	array_view<unsigned int> b3(bytes3.size(), bytes3);
	array_view<unsigned int> b4(bytes4.size(), bytes4);
	array_view<crc16> r(results.size(), results);
	r.discard_data();
	for (int i = 0; i < (65536 / sizeOfLambdaFunction); i++)
	{
		parallel_for_each(e, [=](index<2> idx) restrict(amp, cpu)
			{
				unsigned int initVal = idx[1] + i * sizeOfLambdaFunction;
				unsigned int sum1 = Compute_CRC16_Simple(b1, idx[0], initVal, finalXorVal, resultReflected);
				if (sum1 == resultCRC1 || sum1 == resultCRC1SVOP)
				{
					unsigned int sum2 = Compute_CRC16_Simple(b2, idx[0], initVal, finalXorVal, resultReflected);
					unsigned int sum3 = Compute_CRC16_Simple(b3, idx[0], initVal, finalXorVal, resultReflected);
					unsigned int sum4 = Compute_CRC16_Simple(b4, idx[0], initVal, finalXorVal, resultReflected);
					if (((sum2 == resultCRC2) && (sum3 == resultCRC3) && (sum4 == resultCRC4)) || ((sum2 == resultCRC2SVOP) && (sum3 == resultCRC3SVOP) && (sum4 == resultCRC4SVOP)))
					{
						crc16 result(idx[0], initVal, finalXorVal, inputReflected, resultReflected);
						r[idx[0]] = result;
					}
				}
			});
	}
	r.synchronize();
	for (int i = 0; i < 65536; i++)
	{
		vector<crc16> finalResults;
		if (results.at(i) != crc)
		{
			for (unsigned int initValue = 0x0; initValue <= 0xFFFF; initValue++)
			{
				unsigned int sum1 = Compute_CRC16_Simple(bytes1, results.at(i).p(), initValue, finalXorVal, resultReflected);
				if (sum1 == resultCRC1 || sum1 == resultCRC1SVOP)
				{
					unsigned int sum2 = Compute_CRC16_Simple(bytes2, results.at(i).p(), initValue, finalXorVal, resultReflected);
					unsigned int sum3 = Compute_CRC16_Simple(bytes3, results.at(i).p(), initValue, finalXorVal, resultReflected);
					unsigned int sum4 = Compute_CRC16_Simple(bytes4, results.at(i).p(), initValue, finalXorVal, resultReflected);
					if (((sum2 == resultCRC2) && (sum3 == resultCRC3) && (sum4 == resultCRC4)) || ((sum2 == resultCRC2SVOP) && (sum3 == resultCRC3SVOP) && (sum4 == resultCRC4SVOP)))
					{
						cout << endl << "Найдена комбинация! Polynome: " << hex << uppercase << showbase << results.at(i).p() << " initVal: " << initValue << " finalXorValue: " << finalXorVal << dec << " inputReflected: " << inputReflected << " resultReflected: " << resultReflected << endl;
						crc16 result(results.at(i).p(), initValue, finalXorVal, inputReflected, resultReflected);
						finalResults.push_back(result);
						countResults++;
					}
				}
			}
		}
		if (finalResults.size() != 0)
		{
			writingFile(finalResults, nameOutputFile);
		}
	}
}

void calculatingGPU(unsigned int finalXorValStart)
{
	for (unsigned short int finalXorVal = finalXorValStart; (finalXorVal < 0xFFFF) || (countResults == 0); finalXorVal++)
	{
		eraseFile("Progress.txt");
		writingFile<unsigned int>(finalXorVal, "Progress.txt");
		processing(((finalXorVal == 0xFFFF) ? 0 : (finalXorVal + 1)), 0x10000);
		cout << " finalXorVal: " << hex << uppercase << showbase << setw(6) << finalXorVal;
		calculateCRCwithGPU("1.txt", "2.txt", "3.txt", "4.txt", "results.txt", 0x9420, 0x9732, 0x9A44, 0x9E5C, 0x2094, 0x3297, 0x449A, 0x5C9E, finalXorVal, 0, 0);
		calculateCRCwithGPU("1.txt", "2.txt", "3.txt", "4.txt", "results.txt", 0x9420, 0x9732, 0x9A44, 0x9E5C, 0x2094, 0x3297, 0x449A, 0x5C9E, finalXorVal, 0, 1);
		calculateCRCwithGPU("1.txt", "2.txt", "3.txt", "4.txt", "results.txt", 0x9420, 0x9732, 0x9A44, 0x9E5C, 0x2094, 0x3297, 0x449A, 0x5C9E, finalXorVal, 1, 0);
		calculateCRCwithGPU("1.txt", "2.txt", "3.txt", "4.txt", "results.txt", 0x9420, 0x9732, 0x9A44, 0x9E5C, 0x2094, 0x3297, 0x449A, 0x5C9E, finalXorVal, 1, 1);
	}
	cout << endl;
}

int main()
{
	SetConsoleCP(1251);
	SetConsoleOutputCP(1251);
	try {
		cout << "CRC-16… " << endl;
		while (1)
		{
			vector<unsigned int> xorvalue = loadingFile<unsigned int>("Progress.txt");
			if (xorvalue.size() != 0)
			{
				cout << "Размер передаваемых порций в видеокарту 65536*";
				int sizeOfPartsGPU;
				cin >> sizeOfPartsGPU;
				sizeOfParts = sizeOfPartsGPU;
				calculatingGPU(xorvalue.back());
			}
			else
			{
				cout << "Размер передаваемых порций в видеокарту 65536*";
				int sizeOfPartsGPU;
				cin >> sizeOfPartsGPU;
				sizeOfParts = sizeOfPartsGPU;
				calculatingGPU(0xFFFF);
			}
		}
	}
	catch (runtime_error & e) {
		cerr << "Ошибка! " << e.what() << endl;
	}
	catch (...)
	{
		cerr << "Неизвестная ошибка!" << endl;
	}
	return 0;
}