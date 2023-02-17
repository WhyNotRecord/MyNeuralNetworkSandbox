// MyFirstMLPackApplication.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <mlpack/mlpack.hpp>
#include "FFNTest.h"
#include "RNNTest.h"
#include "WavesPredictor.h"
#include "lstm_stock_prediction.h"


int main()
{
    //std::cout << "Test classification 1\n" << testClassification();
    //std::cout << "Test classification 2\n" << myTestClassification();
    //std::cout << "Test prediction\n" << testPrediction();
   /* for (int i = 10; i < 256; i++)
        std::cout << unsigned char(i);*/
    WavesPredictor wp;
    std::cout << "Test prediction\n" << wp.simplePredict();
    //std::cout << "Test prediction\n" << run_lstm();
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"