---
title: "Topology Optimization Tutorial Report"
author: Zachary Ferguson
date: March 13, 2018
geometry: "left=1in,right=1in,top=1in,bottom=1in"
papersize: letter
fontsize: 10pt
classoption: twocolumn
document-class: article
output: pdf_document
header-includes:
    - \usepackage{diagbox}
    - \usepackage{graphicx}
    - \usepackage{xcolor}
---

# Problem 1

Using the default MMB-beam, Figure \ref{fig:prob_1_rmin} illustrates the affects
of the filter radius, Firgure \ref{fig:prob_1_penal} illustrates the affects
of the penalty power, and Figure \ref{fig:prob_1_discret} illustrates the
affects of the discretization.

## Filter Radius

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_rmin=1_35.pdf}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_rmin=5_4.pdf}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_rmin=10_8.pdf}
\end{center}
\caption{Differing values of rmin.}
\label{fig:prob_1_rmin}
\end{figure}

The filter radius affects how fine the details are. Decreasing the radius
leads to finer structures. Increasing the radius produces softer densities.

## Penalization Power

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_penal=1_5.pdf}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_penal=3.pdf}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_penal=12.pdf}
\end{center}
\caption{Differing values of penal.}
\label{fig:prob_1_penal}
\end{figure}

The penalization power ensures that the solution is black and white. Decreasing
the penalization power will soften the results and increasing will sharpen the
features.

## Discretization

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_ndes=90x30.pdf}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_ndes=180x60.pdf}
\includegraphics[width=.5\textwidth, trim={0, 2cm, 0, 2cm}, clip]{img/problem_1_ndes=360x120.pdf}
\end{center}
\caption{Differing values of (nelx*nely).}
\label{fig:prob_1_discret}
\end{figure}

Decreasing the discretization results in lower resolution results meaning the
features are more soft. Increasing the discretization introduces more grid cells
resulting in sharper more well defined structures.

\newpage
# Problem 2

## Part 1

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_2_1.pdf}
\end{center}
\caption{Two simultaneous point loads.}
\label{fig:prob_2_1}
\end{figure}

## Part 2

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_2_2.pdf}
\end{center}
\caption{Distributed load.}
\label{fig:prob_2_2}
\end{figure}

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_2_2_nonuniform_penal=6.pdf}
\end{center}
\caption{Non-uniform Distributed load.}
\label{fig:prob_2_2}
\end{figure}

\newpage
# Problem 3

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_3_cholmod.pdf}
\end{center}
\caption{Bridge structure with two load cases ({\color{red}red} and
{\color{cyan}cyan} vectors). Solved using CHOLMOD.}
\label{fig:prob_3_cholmod}
\end{figure}

# Problem 4

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_3_nlopt_mma.pdf}
\end{center}
\caption{Bridge structure with two load cases ({\color{red}red} and
{\color{cyan}cyan} vectors). Solved using NLOPT's MMA.}
\label{fig:prob_3_mma}
\end{figure}

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_3_distributed.pdf}
\end{center}
\caption{Bridge structure with multiple load cases. Solved using NLOPT's MMA.}
\label{fig:prob_3_disributed}
\end{figure}

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_3_distributed_nonuniform.pdf}
\end{center}
\caption{Bridge structure with multiple non-uniform load cases. Solved using
NLOPT's MMA.}
\label{fig:prob_3_disributed}
\end{figure}

\newpage
# Problem 5

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_5_rmin=1_4.pdf}
\end{center}
\caption{Compliant mechanism synthesis with a filter radius of 1.4.}
\label{fig:prob_5_1}
\end{figure}

\begin{figure}
\begin{center}
\includegraphics[width=.5\textwidth]{img/problem_5_rmin=3.pdf}
\end{center}
\caption{Compliant mechanism synthesis with a filter radius of 3.}
\label{fig:prob_5_2}
\end{figure}
