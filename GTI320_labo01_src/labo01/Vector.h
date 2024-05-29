#pragma once

/**
 * @file Vector.h
 *
 * @brief Implémentation de vecteurs simples
 *
 * Nom: Benkhalfa Mohamed-Amine
 * Code permanent : BENM77360001
 * Email : mohamed-amine.benkhalfa.1@ens.etsmtl.ca
 *
 */

#include <cmath>
#include "MatrixBase.h"
using namespace std;

namespace gti320 {

	/**
	 * Classe vecteur générique.
	 *
	 * Cette classe réutilise la classe `MatrixBase` et ses spécialisations de
	 * templates pour les manipulation bas niveau.
	 */
	template <typename _Scalar = double, int _Rows = Dynamic>
	class Vector : public MatrixBase<_Scalar, _Rows, 1> {
	public:

		/**
		 * Constructeur par défaut
		 */
		Vector() : MatrixBase<_Scalar, _Rows, 1>() { }

		/**
		 * Contructeur à partir d'un taille (rows).
		 */
		explicit Vector(int rows) : MatrixBase<_Scalar, _Rows, 1>(rows, 1) { }

		/**
		 * Constructeur de copie
		 */
		Vector(const Vector& other) : MatrixBase<_Scalar, _Rows, 1>(other) { }

		/**
		 * Destructeur
		 */
		~Vector() { }

		/**
		 * Opérateur de copie
		 */
		Vector& operator=(const Vector& other)
		{
			// TODO implémenter
			this->m_storage = other.m_storage;
			this->rows = other.rows;
			return *this;
		}

		/**
		 * Accesseur à une entrée du vecteur (lecture seule)
		 */
		_Scalar operator()(int i) const
		{
			// TODO implémenter
			return this->data()[i];
		}

		/**
		 * Accesseur à une entrée du vecteur (lecture et écriture)
		 */
		_Scalar& operator()(int i)
		{
			// TODO implémenter
			return this->m_storage.data()[i];
		}

		/**
		 * Modifie le nombre de lignes du vecteur
		 */
		void resize(int _rows)
		{
			MatrixBase<_Scalar, _Rows, 1>::resize(_rows, 1);
		}

		/**
		 * Produit scalaire de *this et other.
		 */
		inline _Scalar dot(const Vector& other) const
		{
			// TODO implémenter
			_Scalar resultat = 0;
			for (size_t i = 0; i < this->size(); i++)
			{
				resultat += (this->data()[i] * other.data()[i]);
			}
			return resultat;
		}

		/**
		 * Retourne la norme euclidienne du vecteur
		 */
		inline _Scalar norm() const
		{
			// TODO implémenter
			_Scalar resultat = 0;
			for (size_t i = 0; i < this->size(); i++)
			{
				resultat += pow(this->data()[i], 2);
			}
			return sqrt(resultat);
		}
	};
}
