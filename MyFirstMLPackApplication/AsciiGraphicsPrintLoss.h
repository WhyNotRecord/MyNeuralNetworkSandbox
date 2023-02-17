#pragma once

#include <iostream>
#include <mlpack/mlpack.hpp>
#include "ascii_chart.hpp"

const unsigned char UP_TRIANGLE = 30;
const unsigned char DOWN_TRIANGLE = 31;
const unsigned char RIGHT_TRIANGLE = 16;
const unsigned char CIRCLE = 15;

/**
 * Print loss function, based on the EndEpoch callback function, but in pretty way
 */
class AsciiGraphicsPrintLoss
{
public:
    /**
     * Set up the print loss callback class with the width and output stream.
     *
     * @param ostream Ostream which receives output from this object.
     */
    AsciiGraphicsPrintLoss(std::ostream& output = arma::get_cout_stream(), int skipLinesCount = 0)
        : output(output), skippingLines(skipLinesCount)
    { /* Nothing to do here. */
    }

    /**
     * Callback function called at the begin of the optimization process.
     *
     * @param optimizer The optimizer used to update the function.
     * @param function Function to optimize.
     * @param coordinates Starting point.
     */
    template<typename OptimizerType, typename FunctionType, typename MatType>
    void BeginOptimization(OptimizerType& /* optimizer */,
        FunctionType& /* function */,
        MatType& coordinates)
    {
        output << std::endl;
        for (int i = 1; i < 100; i++ ) {
            if (i % 10 == 0) {
                output << i / 10;
            }
            else {
                output << "-";
            }
        }
        output << "0" << std::endl;
    }

    /**
     * Callback function called at the end of a pass over the data.
     *
     * @param optimizer The optimizer used to update the function.
     * @param function Function to optimize.
     * @param coordinates Starting point.
     * @param epoch The index of the current epoch.
     * @param objective Objective value of the current point.
     */
    template<typename OptimizerType, typename FunctionType, typename MatType>
    void EndEpoch(OptimizerType& /* optimizer */,
        FunctionType& /* function */,
        const MatType& /* coordinates */,
        const size_t /* epoch */,
        const double objective)
    {
        if (lastValue == -1) {
            output << CIRCLE;
            //output << RIGHT_TRIANGLE;
        }
        else {//TODO print separate symbols for new minimum and for down move
            if (objective < minValue) {
                output << DOWN_TRIANGLE;
                minValue = objective;
            }
            else if (objective < lastValue) {
                output << RIGHT_TRIANGLE;
            }
            else
            {
                output << UP_TRIANGLE;
            }
            //u8"\u2193" : u8"\u2191" //Unicode down or up arrows, doesn't work
        }
        lastValue = objective;

        values.push_back(objective);
        if (values.size() % 100 == 0)
            output << std::endl;
    }

    /**
     * Callback function called at the end of the optimization process.
     *
     * @param optimizer The optimizer used to update the function.
     * @param function Function to optimize.
     * @param coordinates Starting point.
     */
    template<typename OptimizerType, typename FunctionType, typename MatType>
    void EndOptimization(OptimizerType& optimizer,
        FunctionType& function,
        MatType& coordinates)
    {
        skippingLines = fminl(skippingLines, values.size() / 5);
        output << std::endl;
        if (skippingLines > 0) {
            output << "First " << skippingLines << " loss values skipped are:";
            int prec = output.precision();
            output.precision(5);
            output.setf(std::ios::fixed);
            for (int i = 0; i < skippingLines && i < values.size(); i++) {
                output << " " << values.at(i);
            }
            output << std::endl;
            output.precision(prec);
            output.unsetf(std::ios::fixed);
        }
        while (skippingLines > 0)
        {
            values.erase(values.begin());
            skippingLines--;
        }

        ascii_chart::plot(output, values, 20, fmaxl(values.size() + 5, 100));//TODO certainly make a fixed maximum
        output << std::endl << "Final loss: " << values.at(values.size() - 1) << std::endl;


    }

private:
    //! The output stream that all data is to be sent to; example: std::cout.
    std::ostream& output;
    std::vector<double> values;
    int skippingLines = 0;
    double lastValue = -1.0;
    double minValue = DBL_MAX;
};
