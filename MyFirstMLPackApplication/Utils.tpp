
template<typename eT> void normal_print(std::ostream& o, const arma::Mat<eT> m, bool print_size)
{

    const arma::arma_ostream_state stream_state(o);
    int colWidth = 12;
    if (print_size)
    {
        o.precision(3);
        o.setf(std::ios::fixed, std::ios::floatfield);

        o << "[matrix size: " << m.n_rows << 'x' << m.n_cols << "]\n";
    }

    if (m.n_elem == 0) {
        o.flush();
        stream_state.restore(o);
        return;
    }


    if ((m.n_rows <= 5) && (m.n_cols <= 5)) { arma::arma_ostream::print(o, m, true); return; }

    const bool print_row_ellipsis = (m.n_rows >= 6);
    const bool print_col_ellipsis = (m.n_cols >= 6);

    if ((print_row_ellipsis == true) && (print_col_ellipsis == true))
    {
        arma::Mat<eT> X(4, 4, arma::arma_nozeros_indicator());

        X(arma::span(0, 2), arma::span(0, 2)) = m(arma::span(0, 2), arma::span(0, 2));  // top left submatrix
        X(3, arma::span(0, 2)) = m(m.n_rows - 1, arma::span(0, 2));  // truncated last row
        X(arma::span(0, 2), 3) = m(arma::span(0, 2), m.n_cols - 1);  // truncated last column
        X(3, 3) = m(m.n_rows - 1, m.n_cols - 1);  // bottom right element

        const std::streamsize cell_width = colWidth;

        for (long row = 0; row <= 2; ++row)
        {
            for (long col = 0; col <= 2; ++col)
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }

            o.width(6);
            o << "...";

            o.width(cell_width);
            arma::arma_ostream::print_elem(o, X.at(row, 3), true);
            o << '\n';
        }

        for (long col = 0; col <= 2; ++col)
        {
            o.width(cell_width);
            o << ':';
        }

        o.width(6);
        o << "...";

        o.width(cell_width);
        o << ':' << '\n';

        const long row = 3;
        {
            for (long col = 0; col <= 2; ++col)
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }

            o.width(6);
            o << "...";

            o.width(cell_width);
            arma::arma_ostream::print_elem(o, X.at(row, 3), true);
            o << '\n';
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

            o << '\n';
        }

        for (long col = 0; col < m.n_cols; ++col)
        {
            o.width(cell_width);
            o << ':';
        }

        o.width(cell_width);
        o << '\n';

        const long row = 3;
        {
            for (long col = 0; col < m.n_cols; ++col)
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }
        }

        o << '\n';
    }


    if ((print_row_ellipsis == false) && (print_col_ellipsis == true))
    {
        arma::Mat<eT> X(m.n_rows, 4, arma::arma_nozeros_indicator());

        X(arma::span::all, arma::span(0, 2)) = m(arma::span::all, arma::span(0, 2));  // left
        X(arma::span::all, 3) = m(arma::span::all, m.n_cols - 1);  // right

        const std::streamsize cell_width = colWidth;

        for (long row = 0; row < m.n_rows; ++row)
        {
            for (long col = 0; col <= 2; ++col)
            {
                o.width(cell_width);
                arma::arma_ostream::print_elem(o, X.at(row, col), true);
            }

            o.width(6);
            o << "...";

            o.width(cell_width);
            arma::arma_ostream::print_elem(o, X.at(row, 3), true);
            o << '\n';
        }
    }


    o.flush();
    stream_state.restore(o);
}