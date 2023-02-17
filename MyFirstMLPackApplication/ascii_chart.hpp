/* Signal Epowering Technology   
                       
presents

 █████╗ ███████╗ ██████╗██╗██╗     ██████╗██╗  ██╗ █████╗ ██████╗ ████████╗
██╔══██╗██╔════╝██╔════╝██║██║    ██╔════╝██║  ██║██╔══██╗██╔══██╗╚══██╔══╝
███████║███████╗██║     ██║██║    ██║     ███████║███████║██████╔╝   ██║   
██╔══██║╚════██║██║     ██║██║    ██║     ██╔══██║██╔══██║██╔══██╗   ██║   
██║  ██║███████║╚██████╗██║██║    ╚██████╗██║  ██║██║  ██║██║  ██║   ██║   
╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝╚═╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                                                                             
                                                    Licensed under MPL 2.0. 
                                                   Michael Welsch (c) 2018.
                                                                                                   
a simple function to plot a line to console.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
 
usage: 
ascii_chart::plot(std::vector<T> s0);
ascii_chart::plot(std::vector<T> s0, int heigth, int width);
 
 */

/*** Usage with default L2 metric for abitrary stl containers. ***/

#ifndef ASCII_CHART_H
#define ASCII_CHART_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip> // setprecision
#include <sstream> // stringstream

const unsigned char CROSS = 197;
const unsigned char VERTICAL = 179;
const unsigned char T_CROSS_LEFT = 180;
const unsigned char HORIZONTAL = 196;
const unsigned char LEFT_DOWN = 192;
const unsigned char RIGHT_DOWN = 217;
const unsigned char LEFT_UP = 218;
const unsigned char RIGHT_UP = 191;
const unsigned char SPACE = 32;

namespace ascii_chart
{

template <typename Container>
void plot(std::ostream& output, Container series, int h = 16, int w = 100)
{
    typedef typename Container::value_type T;

    const arma::arma_ostream_state stream_state(output);
    T min = series[0];
    T max = series[0];

    for (int i = 1; i < series.size(); i++)
    {
        min = std::min(min, series[i]);
        max = std::max(max, series[i]);
    }

    T range = std::abs(max - min);

    int offset = 3;
    std::string padding = "           ";

    int height = h;
    T ratio = T(height) / range;
    int min2 = int(std::round(min * ratio));
    int max2 = int(std::round(max * ratio));
    int rows = int(std::abs(max2 - min2));
    int width = w;

    std::vector<std::vector<std::string>> result(rows + 1, std::vector<std::string>(width));

    for (int i = 0; i <= rows; i++)
    {
        for (int j = 0; j < width; j++)
        {
            result[i][j].append(std::string(1, SPACE));
        }
    }
    int precision = 4;

    // axis + labels
    for (int y = min2; y <= max2; ++y)
    {
        float val = float(max) - (float(y) - float(min2)) * float(range) / float(rows);
        std::stringstream stream;
        stream << std::fixed << std::right << std::setw(padding.size()) << std::setprecision(precision) << val;
        std::string slabel = stream.str();

        std::string label = std::string(slabel);
        for (int i = 0; i < result[y - min2][std::max(int(offset) - int(slabel.size()), int(0))].size(); ++i)
        {
            result[y - min2][std::max(int(offset) - int(slabel.size()), int(0))].pop_back();
        }
        result[y - min2][std::max(int(offset) - int(slabel.size()), int(0))].append(slabel);
        while (result[y - min2][std::max(int(offset) - int(slabel.size()), int(0))].size() < padding.size())
        {
            result[y - min2][std::max(int(offset) - int(slabel.size()), int(0))].append(std::string(1, SPACE));
        }

        result[y - min2][offset - 1] = std::string(1, (y == 0) ? CROSS : T_CROSS_LEFT);
    }

    int y0 = int(std::round(series[0] * ratio)) - min2;

    for (int i = 0; i < 4; i++) {
        if (result[rows - y0][offset - 1].size() > 0)
            result[rows - y0][offset - 1].pop_back();
    }
    /*result[rows - y0][offset - 1].pop_back();
    result[rows - y0][offset - 1].pop_back();
    result[rows - y0][offset - 1].pop_back();*/
    result[rows - y0][offset - 1].append(std::string(1, CROSS)); // first value

    for (int x = 0; x < series.size() - 1; x++)
    { // plot the line
        int y0 = int(std::round(series[x + 0] * ratio)) - min2;
        int y1 = int(std::round(series[x + 1] * ratio)) - min2;
        if (y0 == y1)
        {
            result[rows - y0][x + offset] = std::string(1, HORIZONTAL);
        }
        else
        {
            result[rows - y1][x + offset] = std::string(1, (y0 > y1) ? LEFT_DOWN : LEFT_UP);
            result[rows - y0][x + offset] = std::string(1, (y0 > y1) ? RIGHT_UP : RIGHT_DOWN);

            int from = std::min(y0, y1);
            int to = std::max(y0, y1);
            for (int y = from + 1; y < to; y++)
            {
                result[rows - y][x + offset] = std::string(1, VERTICAL);
            }
        }
    }

    for (int i = 0; i < result.size(); ++i)
    {
        for (int j = 0; j < result[i].size(); ++j)
        {
            output << result[i][j];
        }
        output << std::endl;
    }
    output.flush();
    stream_state.restore(output);
}

} // end namespace
#endif