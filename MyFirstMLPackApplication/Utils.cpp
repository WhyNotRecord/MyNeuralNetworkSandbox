#include "Utils.h"
#include <filesystem>

using namespace mlpack;

//TODO print second last column-
template<typename eT> int normal_print(std::ostream& o, const arma::Mat<eT> &m, bool print_size)
{

    const arma::arma_ostream_state stream_state(o);
    eT max = m.max();
    int colWidth = max > 999999999999 ? 15 : 12;
    int ellipsisWidth = 6;

    o.precision(max > 99 ? max > 999999999 ? 1 : 3 : 5);
    o.setf(std::ios::fixed, std::ios::floatfield);

    if (print_size)
    {
        o << "[matrix size: " << m.n_rows << 'x' << m.n_cols << "]" << std::endl;
    }

    if (m.n_elem == 0) {
        o.flush();
        stream_state.restore(o);
        return colWidth;
    }


    if ((m.n_rows <= 5) && (m.n_cols <= 5)) { 
        arma::arma_ostream::print(o, m, true); 
        return colWidth;
    }

    const bool print_row_ellipsis = (m.n_rows >= 6);
    const bool print_col_ellipsis = (m.n_cols >= 6);

    if ((print_row_ellipsis == true) && (print_col_ellipsis == true))
    {
        arma::Mat<eT> X(5, 5, arma::arma_nozeros_indicator());

        X(arma::span(0, 2), arma::span(0, 2)) = m(arma::span(0, 2), arma::span(0, 2));  // top left submatrix
        X(arma::span(3, 4), arma::span(0, 2)) = m(arma::span(m.n_rows - 2, m.n_rows - 1), arma::span(0, 2));  // truncated last row
        X(arma::span(0, 2), arma::span(3, 4)) = m(arma::span(0, 2), arma::span(m.n_cols - 2, m.n_cols - 1));  // truncated last column
        X(3, 3) = m(m.n_rows - 2, m.n_cols - 2);
        X(4, 4) = m(m.n_rows - 1, m.n_cols - 1);  // bottom right element

        const std::streamsize cell_width = colWidth;

        for (long row = 0; row <= 2; ++row)
        {
            for (long col = 0; col <= 2; ++col)//first two cols
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }

            o.width(ellipsisWidth);
            o << "...";

            for (long col = 3; col <= 4; ++col) {// last two cols
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }
            o << std::endl;
        }

        for (long col = 0; col <= 2; ++col)
        {
            o.width(cell_width);
            o << ':';
        }

        o.width(ellipsisWidth);
        o << "...";

        o.width(cell_width);
        o << ':' << std::endl;

        const long row = 3;
        {
            for (long col = 0; col <= 2; ++col)
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }

            o.width(ellipsisWidth);
            o << "...";

            for (long col = 3; col <= 4; ++col) {// last two cols
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }
            o << std::endl;
        }
    }


    if ((print_row_ellipsis == true) && (print_col_ellipsis == false))
    {
        arma::Mat<eT> X(4, m.n_cols, arma::arma_nozeros_indicator());

        X(arma::span(0, 2), arma::span::all) = m(arma::span(0, 2), arma::span::all);  // top
        X(3, arma::span::all) = m(m.n_rows - 1, arma::span::all);  // bottom

        const std::streamsize cell_width = arma::arma_ostream::modify_stream(o, X.memptr(), X.n_elem);

        for (long row = 0; row <= 2; ++row)  // first 3 rows
        {
            for (long col = 0; col < m.n_cols; ++col)
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }

            o << std::endl;
        }

        for (long col = 0; col < m.n_cols; ++col)
        {
            o.width(cell_width);
            o << ':';
        }

        o.width(cell_width);
        o << std::endl;

        const long row = 3;
        {
            for (long col = 0; col < m.n_cols; ++col)
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }
        }

        o << std::endl;
    }


    if ((print_row_ellipsis == false) && (print_col_ellipsis == true))
    {
        arma::Mat<eT> X(m.n_rows, 5, arma::arma_nozeros_indicator());

        X(arma::span::all, arma::span(0, 2)) = m(arma::span::all, arma::span(0, 2));  // left
        X(arma::span::all, arma::span(3, 4)) = m(arma::span::all, arma::span(m.n_cols - 2, m.n_cols - 1));  // right

        const std::streamsize cell_width = colWidth;

        for (long row = 0; row < m.n_rows; ++row)
        {
            for (long col = 0; col <= 2; ++col)
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }

            o.width(ellipsisWidth);
            o << "...";

            for (long col = 3; col <= 4; ++col) {// last two cols
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }
            o << std::endl;
        }
    }


    o.flush();
    stream_state.restore(o);
    return colWidth;
}

template int normal_print<double>(std::ostream& o, const arma::Mat<double> &m, bool print_size);

template<typename eT> void normal_print(std::ostream& o, const arma::Cube<eT>& c, uint16_t slices, float bound, bool print_size) {
    if (slices < 1)
        slices = 1;

    if (slices > c.n_slices)
        slices = c.n_slices;

    if (bound < 0.1f || bound > 0.9f)
        bound = 0.5f;

    const arma::arma_ostream_state stream_state(o);

    if (print_size)
    {
        o << "[cube size: " << c.n_rows << 'x' << c.n_cols << 'x' << c.n_slices << "]" << std::endl;
    }
    int boundSlice = slices * bound;
    int colWidth = 3;

    for (int i = 0; i < boundSlice; i++) {
        o << "Slice " << i;
        if (!print_size)
            o << ":" << std::endl;
        else
            o << " ";
        colWidth = normal_print(o, c.slice(i), print_size);
    }
    if (c.n_slices > slices) {
        for (int i = 0; i < std::fminl(6, c.n_cols); i++) {
            if (c.n_cols > 6 && i == 3) {
                o.width(6);
                o << ":::";
            }
            else {
                o.width(colWidth);
                o << " ::: ";
            }
        }
        o << std::endl;
    }
    for (int i = c.n_slices - (slices - boundSlice); i < c.n_slices; i++) {
        o << "Slice " << i;
        if (!print_size)
            o << ":" << std::endl;
        else
            o << " ";
        normal_print(o, c.slice(i), print_size);
    }

    o.flush();
    stream_state.restore(o);
}

template void normal_print<double>(std::ostream& o, const arma::Cube<double> &c, uint16_t slices, float bound, bool print_size);

std::tuple<arma::mat, arma::mat> loadDataSet(std::string filename, float testDataBound) {
    std::string dataFile = std::string(filename);
    if (!std::filesystem::exists(dataFile)) {
        std::cout << "File " + filename + " not found" << std::endl;
        return {};
    }
    arma::mat trainData;
    data::Load(dataFile, trainData, true);

    int width = trainData.n_cols;
    int fringe = width * testDataBound;
    trainData.shed_cols(fringe, width - 1);

    arma::mat testData;
    data::Load(dataFile, testData, true);
    testData.shed_cols(0, fringe - 1);

    return std::make_tuple(trainData, testData);
}

/*std::tuple<arma::mat, arma::mat> loadDataSetAlt(std::string filename, float testDataBound) {
    std::string dataFile = std::string(filename);
    if (!std::filesystem::exists(dataFile)) {
        std::cout << "File " + filename + " not found" << std::endl;
        return {};
    }
    arma::mat allData;
    data::Load(dataFile, allData, true);

    int width = allData.n_cols;
    int fringe = width * testDataBound;

    arma::mat trainData = allData.submat(0, 0, allData.n_rows - 1, fringe - 1);
    arma::mat testData = allData.submat(0, fringe, allData.n_rows - 1, allData.n_cols - 1);

    return std::make_tuple(trainData, testData);
}*/

std::tuple<arma::mat, arma::mat> loadDataSetAlt(std::string filename, float testDataBound, int crossLen) {
    std::string dataFile = std::string(filename);
    if (!std::filesystem::exists(dataFile)) {
        std::cout << "File " + filename + " not found" << std::endl;
        return {};
    }
    arma::mat allData;
    data::Load(dataFile, allData, true);

    int width = allData.n_cols;
    int fringe = width * testDataBound;
    int fringeCross = fringe + crossLen;
    if (fringeCross >= width)
        fringeCross = fringe;

    arma::mat trainData = allData.submat(0, 0, allData.n_rows - 1, fringeCross - 1);
    arma::mat testData = allData.submat(0, fringe, allData.n_rows - 1, allData.n_cols - 1);

    return std::make_tuple(trainData, testData);
}


std::tuple<arma::cube, arma::cube> loadDataSetCube(std::string filename, float testDataBound) {
    std::string dataFile = std::string(filename);
    if (!std::filesystem::exists(dataFile)) {
        std::cout << "File " + filename + " not found" << std::endl;
        return {};
    }
    arma::mat allDataMat;
    data::Load(dataFile, allDataMat, true);

    int width = allDataMat.n_cols;
    int fringe = width * testDataBound;

    arma::cube trainData(allDataMat.n_rows, fringe, 1);
    arma::mat trdm = allDataMat.submat(0, 0, allDataMat.n_rows - 1, fringe - 1);
    trainData.slice(0) = trdm;
    arma::cube testData(allDataMat.n_rows, width - fringe, 1);
    arma::mat tdm = allDataMat.submat(0, fringe, allDataMat.n_rows - 1, allDataMat.n_cols - 1);
    testData.slice(0) = tdm;

    return std::make_tuple(trainData, testData);
}

std::tuple<arma::cube, arma::cube> preparePredictionCubes(arma::mat &valuesMatrix, int analCount, int predictCount, bool print) {
    size_t numColumns = valuesMatrix.n_cols - analCount - predictCount;
    //row-col-slice
    arma::cube data = arma::cube(1, numColumns, analCount);
    arma::cube labels = arma::cube(predictCount, numColumns, 1);

    for (size_t i = 0; i < numColumns; ++i)
    {
        if (print)
            std::cout << "column " << i << std::endl;

        //auto dataVec = valuesMatrix.cols(i, i + analCount - 1);
        data.tube(0, i) = valuesMatrix.cols(i, i + analCount - 1);
        if (print)
            data.tube(0, i).brief_print(std::cout, "");

        //auto labelsVec = valuesMatrix.cols(i + analCount, i + analCount + predictCount - 1);
        /*labels.tube(0, i) = valuesMatrix.cols(i + analCount, i + analCount + predictCount - 1);
        if (print)
            labels.tube(0, i).brief_print(std::cout, "");*/
        auto labelsVec = valuesMatrix.cols(i + analCount, i + analCount + predictCount - 1);
        auto labelsVecTrans = labelsVec.as_col();
        //labels.col(i) = labelsVecTrans;
        labels.subcube(arma::span(), arma::span(i), arma::span()) = labelsVecTrans;
            //valuesMatrix.submat(arma::span(3, 4), arma::span(0));

    }
    return std::make_tuple(data, labels);
}

std::tuple<arma::cube, arma::cube> preparePredictionCubes(arma::mat &valuesMatrix, int inputCount, int analCount, int predictCount, bool print) {
    size_t numColumns = valuesMatrix.n_cols - analCount - predictCount;
    //row-col-slice
    arma::cube data = arma::cube(inputCount, numColumns, analCount);
    arma::cube labels = arma::cube(predictCount, numColumns, analCount);

    for (size_t i = 0; i < numColumns; i++)
    {
        data.subcube(arma::span(), arma::span(i), arma::span()) =
            valuesMatrix.submat(arma::span(), arma::span(i, i + analCount - 1));
        auto labelsMat = valuesMatrix.submat(arma::span(0, 1), arma::span(i + 1, i + analCount));
        labels.subcube(arma::span(), arma::span(i), arma::span()) = labelsMat;
    }
    return std::make_tuple(data, labels);
}


std::array<double, 3> calculateDifferencePrecents(arma::cube& control, arma::cube& checking) {
    if (control.n_rows != checking.n_rows || control.n_cols != checking.n_cols || control.n_slices != checking.n_slices) {
        std::cout << "ERROR: Cubes are of different sizes!" << std::endl;
        return std::array<double, 3>();
    }
    std::array<double, 3> result = { 1000., 0., 0. };
    for (int r = 0; r < control.n_rows; r++) {
        for (int c = 0; c < control.n_cols; c++) {
            for (int s = 0; s < control.n_slices; s++) {
                /*double con = control(r, c, s), che = checking(r, c, s);
                double tmp = con / 100.f;
                tmp = std::fabsf(con - che) / tmp;
                result[1] += tmp;*/
                double tmp = control(r, c, s) / 100.f;
                tmp = std::fabsf(control(r, c, s) - checking(r, c, s)) / tmp;
                result[1] += tmp;
                if (result[0] > tmp)
                    result[0] = tmp;
                if (result[2] < tmp)
                    result[2] = tmp;
            }
        }
    }
    result[1] /= control.n_elem;
    return result;
}

//template<typename eT> std::array<double, 3> calculateDifferencePrecents(arma::subview_cube<eT>& control, arma::subview_cube<eT>& checking) {
std::array<double, 3> calculateDifferencePrecents(arma::subview_cube<double> control, arma::subview_cube<double> checking) {
    if (control.n_rows != checking.n_rows || control.n_cols != checking.n_cols || control.n_slices != checking.n_slices) {
        std::cout << "ERROR: Cubes are of different sizes!" << std::endl;
        return std::array<double, 3>();
    }
    std::array<double, 3> result = { 1000., 0., 0. };
    for (int r = 0; r < control.n_rows; r++) {
        for (int c = 0; c < control.n_cols; c++) {
            for (int s = 0; s < control.n_slices; s++) {
                /*double con = control(r, c, s), che = checking(r, c, s);
                double tmp = con / 100.f;
                tmp = std::fabsf(con - che) / tmp;
                result[1] += tmp;*/
                double tmp = control(r, c, s) / 100.f;
                tmp = std::fabsf(control(r, c, s) - checking(r, c, s)) / tmp;
                result[1] += tmp;
                if (result[0] > tmp)
                    result[0] = tmp;
                if (result[2] < tmp)
                    result[2] = tmp;
            }
        }
    }
    result[1] /= control.n_elem;
    return result;
}

//template std::array<double, 3> calculateDifferencePrecents<double>(arma::subview_cube<double>& control, arma::subview_cube<double>& checking);


std::array<double, 3> calculateDifferencePrecents(arma::mat& control, arma::mat& checking) {
    if (control.n_rows != checking.n_rows || control.n_cols != checking.n_cols) {
        std::cout << "ERROR: Matrices are of different sizes!" << std::endl;
        return std::array<double, 3>();
    }
    std::array<double, 3> result = { 1000., 0., 0. };
    for (int r = 0; r < control.n_rows; r++) {
        for (int c = 0; c < control.n_cols; c++) {
            double tmp = std::fabsf(control(r, c) - checking(r, c)) / control(r, c) * 100.f;
            /*double tmp = control(r, c) / 100.f;
            tmp = std::fabsf(control(r, c) - checking(r, c)) / tmp;*/
            result[1] += tmp;
            if (result[0] > tmp)
                result[0] = tmp;
            if (result[2] < tmp)
                result[2] = tmp;
        }
    }
    result[1] /= control.n_elem;
    return result;
}

/*
 * Function to calculate MSE for arma::cube.
 */
double ComputeMSE(arma::cube& pred, arma::cube& Y)
{
    return metric::SquaredEuclideanDistance::Evaluate(pred, Y) / (Y.n_elem);
}

FFN<> loadModel(std::string filename) {
    FFN<> model;
    data::Load(filename, "model", model);
    return model;
}

int saveModel(FFN<> model, std::string name) {
    if (!data::Save(name + ".xml", name, model, false))
        return 1;
    return 0;
}


