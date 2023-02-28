function svd2inv(M)

	X = svd(M)
	Minv = X.Vt' * Diagonal(1 ./ X.S) * X.U'
	Minv = (Minv + Minv')/2

	return Minv 
end