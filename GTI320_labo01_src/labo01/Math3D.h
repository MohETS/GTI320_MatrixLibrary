#pragma once

/**
 * @file Math3D.h
 *
 * @brief Fonctions pour l'intinialisation et la manipulation de matrices de
 * rotation, des matrices de transformations en coordonnées homogènes et les
 * vecteurs 3D.
 *
 * Nom: Benkhalfa Mohamed-Amine
 * Code permanent : BENM77360001
 * Email : mohamed-amine.benkhalfa.1@ens.etsmtl.ca
 *
 */

#include "Matrix.h"
#include "Vector.h"
#include "Operators.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>


namespace gti320 {

	// Deux types de vecteurs 3D considérés ici
	typedef Vector<double, 3> Vector3d;
	typedef Vector<float, 3> Vector3f;

	// Dans le cadre de ce projet, nous considérons seulement deux
	// cas :
	//
	//  - les rotations
	//  - les translations
	//
	// Deux types de matrices en coordonnées homogèes :
	typedef Matrix<double, 4, 4, ColumnStorage> Matrix4d;
	typedef Matrix<float, 4, 4, ColumnStorage> Matrix4f;
	// 
	// Deux types de matrices pour les rotations
	typedef Matrix<double, 3, 3, ColumnStorage> Matrix3d;
	typedef Matrix<float, 3, 3, ColumnStorage> Matrix3f;

	/**
	 * Initialise et retourne la matrice identité
	 */
	template<>
	inline void Matrix4d::setIdentity()
	{
		// TODO affecter la valeur 0.0 partout, sauf sur la diagonale principale où c'est 1.0.
		//      Note: ceci est une redéfinition d'une fonction membre!
		memset(m_storage.data(), 0, size() * sizeof(double));
		for (size_t i = 0; i < this->cols(); ++i)
		{
			m_storage.data()[(i * this->cols()) + i] = 1;
		}
	}

	/**
	 * Calcul de la matrice inverse, SPÉCIALISÉ pour le cas d'une matrice de
	 * transformation en coordonnées homogènes.
	 *
	 * TODO (vous pouvez supposer qu'il s'agit d'une matrice de transformation
	 * en coordonnées homogènes)
	 */
	template<>
	inline Matrix4d Matrix4d::inverse() const
	{
		// TODO : implémenter
		Matrix4d inversedMatrix;
		inversedMatrix.setZero();

		for (size_t j = 0; j < 3; ++j)
		{
			for (size_t i = 0; i < 3; ++i)
			{
				inversedMatrix(i, j) = data()[(i * 4) + j];
				inversedMatrix(i, 3) += inversedMatrix(i, j) * -data()[(3 * 4) + j];
				
			}
		}
		inversedMatrix(3, 3) = 1;

		return inversedMatrix;
		//return Matrix4d(); // Pas bon, à changer
	}

	/**
	 * Calcul de la matrice inverse, SPÉCIALISÉ pour le cas d'une matrice de rotation.
	 *
	 * (vous pouvez supposer qu'il s'agit d'une matrice de rotation)
	 */
	template<>
	inline Matrix3d Matrix3d::inverse() const
	{
		// TODO : implémenter
		Matrix3d inversedMatrix;
		for (size_t j = 0; j < 3; ++j)
		{
			for (size_t i = 0; i < 3; ++i)
			{
				inversedMatrix(i, j) = data()[(i * 3) + j];
			}
		}
		return inversedMatrix;
	}


	/**
	 * Multiplication d'une matrice 4x4 avec un vecteur 3D où la matrice
	 * représente une transformation en coordonnées homogène.
	 */
	template <typename _Scalar>
	Vector<_Scalar, 3> operator*(const Matrix<_Scalar, 4, 4, ColumnStorage>& A, const Vector<_Scalar, 3>& v)
	{
		// TODO : implémenter
		Vector<_Scalar, 3> newVector;
		newVector.setZero();
		_Scalar u[4] = { v(0),v(1),v(2),1 };

		for (size_t j = 0; j < 4; ++j)
		{
			//_Scalar u = v(j);
			for (size_t i = 0; i < 3; ++i)
			{
				newVector(i) += A(i, j) * u[j];
			}
		}
		return newVector;
		//return Vector<_Scalar, 3>(); // pas bon, à changer
	}


	/**
	 * Initialise et retourne la matrice de rotation définie par les angles
	 * d'Euler XYZ exprimés en radians.
	 *
	 * La matrice doit correspondre au produit : Rz*Ry*Rx.
	 */
	template<typename _Scalar>
	static Matrix<_Scalar, 3, 3> makeRotation(_Scalar x, _Scalar y, _Scalar z)
	{
		// TODO : implémenter
		Matrix3d rotationMatrix;

		double cosX = cos(x), sinX = sin(x);
		double cosY = cos(y), sinY = sin(y);
		double cosZ = cos(z), sinZ = sin(z);

		//Premiere Colonne
		rotationMatrix(0, 0) = cosY * cosZ;
		rotationMatrix(1, 0) = cosY * sinZ;
		rotationMatrix(2, 0) = -sinY;


		//Deuxieme Colonne
		rotationMatrix(0, 1) = (sinX * sinY * cosZ) - (cosX * sinZ);
		rotationMatrix(1, 1) = (sinX * sinY * sinZ) + (cosX * cosZ);
		rotationMatrix(2, 1) = sinX * cosY;

		//Troisieme Colonne
		rotationMatrix(0, 2) = (cosX * sinY * cosZ) + (sinX * sinZ);
		rotationMatrix(1, 2) = (cosX * sinY * sinZ) - (sinX * cosZ);
		rotationMatrix(2, 2) = cosX * cosY;

		return rotationMatrix;
	}

}
