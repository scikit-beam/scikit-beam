//for Numberical recipes, modified for ints
//updated from my version by online version. June/06
void hunt(long int xx[], unsigned long n, long int x, unsigned long *jlo)
{
	unsigned long jm,jhi,inc;
	int ascnd;

	ascnd=(xx[n] >= xx[1]);
	if (*jlo <= 0 || *jlo > n) {
		*jlo=0;
		jhi=n+1;
	} else {
		inc=1;
		if ((x >= xx[*jlo]) == ascnd) {
			if (*jlo == n) return;
			jhi=(*jlo)+1;
			while ((x >= xx[jhi]) == ascnd) {
				*jlo=jhi;
				inc += inc;
				jhi=(*jlo)+inc;
				if (jhi > n) {
					jhi=n+1;
					break;
				}
			}
		} else {
			if (*jlo == 1) {
				*jlo=0;
				return;
			}
			jhi=(*jlo)--;
			while ((x < xx[*jlo]) == ascnd) {
				jhi=(*jlo);
				inc <<= 1;
				if (inc >= jhi) {
					*jlo=0;
					break;
				}
				else *jlo=jhi-inc;
			}
		}
	}
	while (jhi-(*jlo) != 1) {
		jm=(jhi+(*jlo)) >> 1;
		if ((x >= xx[jm]) == ascnd)
			*jlo=jm;
		else
			jhi=jm;
	}
  if (x == xx[n]) *jlo=n;// special for droplet
  if (x == xx[1]) *jlo=1;
}
/* (C) Copr. 1986-92 Numerical Recipes Software ^. */
