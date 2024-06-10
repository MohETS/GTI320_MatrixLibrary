#pragma once

/**
 * @file Operators.h
 *
 * @brief Opérateurs arithmétiques pour les matrices et les vecteurs.
 *
 * Nom: Benkhalfa Mohamed-Amine
 * Code permanent : BENM77360001
 * Email : mohamed-amine.benkhalfa.1@ens.etsmtl.ca
 *
 */

#include "Matrix.h"
#include "Vector.h"

 /**
  * Implémentation de divers opérateurs arithmétiques pour les matrices et les vecteurs.
  */
namespace gti320 {

	/**
	 * Multiplication : Matrice * Matrice (générique)
	 */
	template <typename _Scalar, int RowsA, int ColsA, int StorageA, int RowsB, int ColsB, int StorageB>
	Matrix<_Scalar, RowsA, ColsB> operator*(const Matrix<_Scalar, RowsA, ColsA, StorageA>& A, const Matrix<_Scalar, RowsB, ColsB, StorageB>& B)
	{
		// TODO implémenter
		int newMatrixRows = A.rows(), newMatrixCols = B.cols();
		Matrix<_Scalar, Dynamic, Dynamic> newMatrix(newMatrixRows, newMatrixCols);

		for (size_t j = 0; j < newMatrixCols; ++j)
		{
			for (size_t i = 0; i < newMatrixRows; ++i)
			{
				_Scalar value = B(i, j);
				for (int k = 0; k < newMatrixCols; ++k)
				{
					newMatrix(k, j) += A(k, i) * value;
				}
			}
		}
		return newMatrix;
	}

	/**
	 * Multiplication : Matrice (colonne) * Matrice (ligne)
	 *
	 * Spécialisation de l'opérateur de multiplication pour le cas où les matrices
	 * ont un stockage à taille dynamique et où la matrice de gauche utilise un
	 * stockage par colonnes et celle de droite un stockage par lignes.
	 */
	template <typename _Scalar>
	Matrix<_Scalar, Dynamic, Dynamic> operator*(const Matrix<_Scalar, Dynamic, Dynamic, ColumnStorage>& A, const Matrix<_Scalar, Dynamic, Dynamic, RowStorage>& B)
	{
		// TODO : implémenter
		int newMatrixRows = A.rows(), newMatrixCols = B.cols();
		Matrix<_Scalar, Dynamic, Dynamic> newMatrix(newMatrixRows, newMatrixCols);

		for (size_t j = 0; j < newMatrixCols; ++j)
		{
			for (size_t i = 0; i < newMatrixRows; ++i)
			{
				_Scalar value = B(i, j);
				for (int k = 0; k < newMatrixCols; ++k)
				{
					newMatrix(k, j) += A(k, i) * value;
				}
			}
		}
		return newMatrix;
	}

	/**
	 * Multiplication : Matrice (ligne) * Matrice (colonne)
	 *
	 * Spécialisation de l'opérateur de multiplication pour le cas où les matrices
	 * ont un stockage à taille dynamique et où la matrice de gauche utilise un
	 * stockage par lignes et celle de droite un stockage par colonnes.
	 */
	template <typename _Scalar>
	Matrix<_Scalar, Dynamic, Dynamic> operator*(const Matrix<_Scalar, Dynamic, Dynamic, RowStorage>& A, const Matrix<_Scalar, Dynamic, Dynamic, ColumnStorage>& B)
	{
		// TODO : implémenter
		int newMatrixRows = A.rows(), newMatrixCols = B.cols();
		Matrix<_Scalar, Dynamic, Dynamic> newMatrix(newMatrixRows, newMatrixCols);

		/*for (size_t i = 0; i < newMatrixRows; ++i)
		{
			for (size_t j = 0; j < newMatrixCols; ++j)
			{
				for (int k = 0; k < newMatrixCols; ++k)
				{
					newMatrix(i, j) += A(i, k) * B(k, j);
				}
			}
		}
		return newMatrix;*/

		for (size_t i = 0; i < newMatrixCols; ++i)
		{
			for (size_t j = 0; j < newMatrixRows; ++j)
			{
				_Scalar value = B(i, j);
				for (int k = 0; k < newMatrixCols; ++k)
				{
					newMatrix(k, j) += A(k, i) * value;
				}
			}
		}
		return newMatrix;
	}


	/**
	 * Addition : Matrice + Matrice (générique)
	 */
	template <typename _Scalar, int Rows, int Cols, int StorageA, int StorageB>
	Matrix<_Scalar, Rows, Cols> operator+(const Matrix<_Scalar, Rows, Cols, StorageA>& A, const Matrix<_Scalar, Rows, Cols, StorageB>& B)
	{
		// TODO : implémenter
		int newMatrixRows = A.rows(), newMatrixCols = A.cols();
		Matrix<_Scalar, Rows, Cols> newMatrix(newMatrixRows, newMatrixCols);


		for (size_t j = 0; j < newMatrixCols; ++j)
		{
			for (size_t i = 0; i < newMatrixRows; ++i)
			{
				newMatrix(i, j) = A(i, j) + B(i, j);
			}
		}
		return newMatrix;
	}

	/**
	 * Addition : Matrice (colonne) + Matrice (colonne)
	 *
	 * Spécialisation de l'opérateur d'addition pour le cas où les deux matrices
	 * sont stockées par colonnes.
	 */
	template <typename _Scalar>
	Matrix<_Scalar, Dynamic, Dynamic> operator+(const Matrix<_Scalar, Dynamic, Dynamic, ColumnStorage>& A, const Matrix<_Scalar, Dynamic, Dynamic, ColumnStorage>& B)
	{
		// TODO : implémenter
		Matrix<_Scalar, Dynamic, Dynamic, ColumnStorage> newMatrix(A);
		int newMatrixRows = newMatrix.rows(), newMatrixCols = newMatrix.cols();

		for (size_t j = 0; j < newMatrixCols; ++j)
		{
			for (size_t i = 0; i < newMatrixRows; ++i)
			{
				newMatrix(i, j) += B(i, j);
			}
		}
		return newMatrix;
	}

	/**
	 * Addition : Matrice (ligne) + Matrice (ligne)
	 *
	 * Spécialisation de l'opérateur d'addition pour le cas où les deux matrices
	 * sont stockées par lignes.
	 */
	template <typename _Scalar>
	Matrix<_Scalar, Dynamic, Dynamic, RowStorage> operator+(const Matrix<_Scalar, Dynamic, Dynamic, RowStorage>& A, const Matrix<_Scalar, Dynamic, Dynamic, RowStorage>& B)
	{
		// TODO : implémenter
		Matrix<_Scalar, Dynamic, Dynamic, RowStorage> newMatrix(A);
		int newMatrixRows = newMatrix.rows(), newMatrixCols = newMatrix.cols();

		for (size_t i = 0; i < newMatrixRows; ++i)
		{
			for (size_t j = 0; j < newMatrixCols; ++j)
			{
				newMatrix(i, j) += B(i, j);
			}
		}
		return newMatrix;
	}

	/**
	 * Multiplication  : Scalaire * Matrice (colonne)
	 *
	 * Spécialisation de l'opérateur de multiplication par un scalaire pour le
	 * cas d'une matrice stockée par colonnes.
	 */
	template <typename _Scalar, int _Rows, int _Cols>
	Matrix<_Scalar, _Rows, _Cols, ColumnStorage> operator*(const _Scalar& a, const Matrix<_Scalar, _Rows, _Cols, ColumnStorage>& A)
	{
		// TODO : implémenter
		Matrix<_Scalar, _Rows, _Cols, ColumnStorage> newMatrix(A);
		int newMatrixRows = newMatrix.rows(), newMatrixCols = newMatrix.cols();

		for (size_t j = 0; j < newMatrixCols; ++j)
		{
			for (size_t i = 0; i < newMatrixRows; ++i)
			{
				newMatrix(i, j) *= a;
			}
		}
		return newMatrix;
	}

	/**
	 * Multiplication  : Scalaire * Matrice (ligne)
	 *
	 * Spécialisation de l'opérateur de multiplication par un scalaire pour le
	 * cas d'une matrice stockée par lignes.
	 */
	template <typename _Scalar, int _Rows, int _Cols>
	Matrix<_Scalar, _Rows, _Cols, RowStorage> operator*(const _Scalar& a, const Matrix<_Scalar, _Rows, _Cols, RowStorage>& A)
	{
		// TODO : implémenter
		Matrix<_Scalar, Dynamic, Dynamic, RowStorage> newMatrix(A);
		int newMatrixRows = newMatrix.rows(), newMatrixCols = newMatrix.cols();

		for (size_t i = 0; i < newMatrixRows; ++i)
		{
			for (size_t j = 0; j < newMatrixCols; ++j)
			{
				newMatrix(i, j) *= a;
			}
		}
		return newMatrix;
	}

	/**
	 * Multiplication : Matrice (ligne) * Vecteur
	 *
	 * Spécialisation de l'opérateur de multiplication matrice*vecteur pour le
	 * cas où la matrice est représentée par lignes.
	 */
	template <typename _Scalar, int _Rows, int _Cols>
	Vector<_Scalar, _Rows> operator*(const Matrix<_Scalar, _Rows, _Cols, RowStorage>& A, const Vector<_Scalar, _Cols>& v)
	{
		// TODO : implémenter
		int matrixRows = A.rows(), matrixCols = A.cols();
		Vector<_Scalar, _Rows> newVector(matrixRows);

		for (size_t i = 0; i < matrixRows; ++i)
		{
			for (size_t j = 0; j < matrixCols; ++j)
			{
				newVector(i) += A(i, j) * v(j);;
			}
		}

		return newVector;
	}

	/**
	 * Multiplication : Matrice (colonne) * Vecteur
	 *
	 * Spécialisation de l'opérateur de multiplication matrice*vecteur pour le
	 * cas où la matrice est représentée par colonnes.
	 */
	template <typename _Scalar, int _Rows, int _Cols>
	Vector<_Scalar, _Rows> operator*(const Matrix<_Scalar, _Rows, _Cols, ColumnStorage>& A, const Vector<_Scalar, _Cols>& v)
	{
		// TODO : implémenter
		int matrixRows = A.rows(), matrixCols = A.cols();
		Vector<_Scalar, _Rows> newVector(matrixRows);

		for (size_t j = 0; j < matrixCols; ++j)
		{
			_Scalar u = v(j);
			for (size_t i = 0; i < matrixRows; ++i)
			{
				newVector(i) += A(i, j) * u;
			}
		}

		return newVector;
	}

	/**
	 * Multiplication : Scalaire * Vecteur
	 */
	template <typename _Scalar, int _Rows>
	Vector<_Scalar, _Rows> operator*(const _Scalar& a, const Vector<_Scalar, _Rows>& v)
	{
		// TODO : implémenter
		Vector<_Scalar, _Rows> newVector(v);

		for (size_t i = 0; i < v.size(); ++i)
		{
			newVector(i) *= a;
		}

		return newVector;
	}


	/**
	 * Addition : Vecteur + Vecteur
	 */
	template <typename _Scalar, int _RowsA, int _RowsB>
	Vector<_Scalar, _RowsA> operator+(const Vector<_Scalar, _RowsA>& a, const Vector<_Scalar, _RowsB>& b)
	{
		// TODO : implémenter
		Vector<_Scalar, _RowsA> newVector(a);

		for (size_t i = 0; i < a.size(); ++i)
		{
			newVector(i) += b(i);
		}

		return newVector;
	}

	/**
	 * Soustraction : Vecteur - Vecteur
	 */
	template <typename _Scalar, int _RowsA, int _RowsB>
	Vector<_Scalar, _RowsA> operator-(const Vector<_Scalar, _RowsA>& a, const Vector<_Scalar, _RowsB>& b)
	{
		// TODO : implémenter
		Vector<_Scalar, _RowsA> newVector(a);

		for (size_t i = 0; i < a.size(); ++i)
		{
			newVector(i) -= b(i);
		}

		return newVector;
	}
}
