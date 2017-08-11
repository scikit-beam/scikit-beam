/* orginal from Larry Lurio (July 2005). */
/* modified  Mark Sutton (June 2006) 
 * rewrote peakfind to handle arbitrarily complicated droplets
 * changed img_anal to handle new ids
 * Change names to dropletfind and dropletanal
 */
#include <stdio.h>
#include <stdlib.h>
#include "hunt.c"


extern void raw_expand(long int *img_out, long int *img_in, int ncol, int nrow);

void raw_expand(long int *img_out, long int *img_in, int ncol, int nrow)
{
	//This is not always correct, seems to be problems at edges only.
	int i;
	for (i=0;i<nrow*ncol;i++) {
		if (img_in[i]) {
			if (i>ncol) {
				if (i%ncol>1) img_out[i-ncol-1] = 1;
				img_out[i-ncol] = 1;
				if (i%ncol<ncol-1) img_out[i-ncol+1] = 1;
			        }
			if (i%ncol>1) img_out[i-1] = 1;
			img_out[i] = 1;
			if (i%ncol<ncol-1) img_out[i+1] = 1;
			if (i < (nrow-1)*ncol) {
				if (i%ncol>1) img_out[i+ncol-1] = 1;
				img_out[i+ncol] = 1;
				if (i%ncol<ncol-1) img_out[i+ncol+1] = 1;
			        }
			}
		}
}



extern void raw_dropletanal(long int *img, long int *bimg, long int *npix,
        double *xcen, double *ycen, long int *adus, long int *idlist, int npeak,int
        ncol, int nrow);

//Should adus vector be float or double?
void raw_dropletanal(long int *img, long int *bimg, long int *npix, double 
        *xcen, double *ycen, long int *adus,long int *idlist, int npeak,int
        ncol, int nrow) {
	int i,id;
    unsigned long int pnum, pos;
	float x,y;
	pnum=0; //size of idlist
	for (i=0;i<nrow*ncol;i++) {
		if (bimg[i]) { 
			id=bimg[i];
			if(id!=i+1){ // make cycles to traverse droplets
				bimg[i]=bimg[id-1];//swap ptrs
				bimg[id-1]=i+1;
				}
			if(pnum==0 || id>idlist[pnum-1]) { //must be if a new droplet.
			    idlist[pnum]=id;
			    pnum++;
			    pos=pnum;
			    //printf("add %d,%d,%d,%d\n",id,pos,idlist[pos-1],idlist[pos]);
			    }	
			if( id!= idlist[pos-1]) {
				hunt(idlist-1,pnum,id,&pos);
			        //printf("hunt %d,%d,%d,%d\n",id,pos,idlist[pos-1],idlist[pos]);
				if(idlist[pos-1]!=id) {
				    //I don't think this should happen and I
				    //will remove this at some point if it
				    //doesn't happen in practice.
				    //If it does happen, figure out why and fix.
				    printf("error should not be here, not in list\n");
			    printf("err %d,%ld,%ld,%ld,%ld\n",id,pos,idlist[pos-1],idlist[pos],idlist[pnum-1]);
				    idlist[pnum]=id; // this is proabably wrong
				    pnum++;
				    pos=pnum;
				    }	
				}
			x=(double)(i%ncol);
			y=(double)(i/ncol);
			npix[pos-1]+=1;
			xcen[pos-1]+=x*(double)(img[i]);
			ycen[pos-1]+=y*(double)(img[i]);
			adus[pos-1]+=img[i];
			}
		}
	for (i=0;i<npeak;i++) {
		if (adus[i]) {
			xcen[i]/=(double)(adus[i]);
			ycen[i]/=(double)(adus[i]);
			}
		else {
			printf("bad droplet for i=%d\n",i);
			printf("xcen=%lf ycen=%lf adus=%ld npix=%ld\n",xcen[i],ycen[i],adus[i],npix[i]);
			}
	    }
}

extern void raw_dropletfind(long int *img_out, long int *img_in, int ncol, int nrow, long int *npeak);

void raw_dropletfind(long int *img_out,long int *img_in, int ncol, int nrow, long int *npeak) {
/* As we sweep through the bitmap check left and above points to see
 * if part of a droplet.
 * Trick is to use the index of the "first" occurence of a pixel in
 * a droplet as the droplet id (plus 1 because background will be droplet
 * 0]. Then each added pixel to a droplet will
 * get its id. This allows for a straightforward merging of complicated
 * (hairy-like) droplets with a not much looping over droplet ids.
 */
	int i,j,curr,above,left,peakcount,complicated;
	int typ,oldtyp,pos,pos1,pos2;
        complicated=0;
	//Have special cases of 1)first pixel      2)rest of first row
	//		        3)start of columns 4) interior pixels
	//First Pixel
	if(img_in[0]) img_out[0]=1;
	peakcount=img_out[0];
	//Rest of first row
	for(i=1;i<ncol;i++) { /* id first row */
	    if(img_in[i]) {
		if(img_in[i-1]) img_out[i]=img_out[i-1];
		else {
		    img_out[i]=i+1;
		    peakcount++;
		    }
		}
	    }
	/* matrix layout(in C) [[0,1,2,...,ncol-1]
	 *                      [ncol,...,2ncol-1]
	 *                      ...
	 *                      [(nrow-1)ncol,...,nrow*ncol-1]]
	 * so above of i is i-ncol and left of i is i-1
	 */
	curr=ncol; //current pixel to study
	above=0;
	left=ncol-1;
	for (i=1;i<nrow;i++) {
	    oldtyp=0;
	    //Check first col
	    if(img_in[curr]) {
		    if(img_in[above]) {
			img_out[curr]=img_out[above];
			oldtyp=3;
			}
		    else {
			img_out[curr]=curr+1;
			peakcount++;
			oldtyp=1;
			}
		    }
	       above++; curr++; left++;
	    //Interior pixels
	    for (j=1;j<ncol;j++) {
		typ=0;
		    //printf("%d,%d,%d,%d,%d,%d,%d\n",i,j,typ,curr,above,left,img_in[curr]);
		if (img_in[curr]) {
		    typ=1+img_in[above]*2+img_in[left];
		    //printf("---%d,%d,%d,%d,%d\n",curr,typ,img_in[curr],img_in[above],img_in[left]);
		    /* Only four possibilities for left and above */
		    switch(typ){
			case 1: img_out[curr]=curr+1;
			   //printf("new id %d,%d,%d\n",curr,typ,oldtyp);
			   peakcount++; /*new droplet */
			   break;
			case 2: img_out[curr]=img_out[left];
			   break;
			case 3: img_out[curr]=img_out[above];
			   break;
			case 4: img_out[curr]=img_out[above];
			   //printf("merge %d,%d,%d\n",curr,typ,oldtyp);
			   if(oldtyp==1) {
				img_out[left]=img_out[curr];
			        peakcount--; /* simple merge of two droplets */
			        }
			   if(oldtyp==2) { /*oops complicated */
			       //printf("complicated\n");
			       complicated=1;
			       pos=img_out[left]-1; //find left's minimum id
			       while(img_out[pos]!=pos+1) pos=img_out[pos]-1;
			       pos1=pos;
			       pos=img_out[above]-1; //find above's minimum id
			       while(img_out[pos]!=pos+1) pos=img_out[pos]-1;
			       pos2=pos;
			       if(pos1<pos) pos=pos1; //most minimum, helps img_anal
			       //printf("pos1,pos2 %d,%d,%d\n",pos,pos1,pos2);
			       //id them all
			       img_out[curr]=img_out[pos]; //add new pixel
			       img_out[pos1]=img_out[pos]; //merge droplets
			       img_out[pos2]=img_out[pos];
			       img_out[left]=img_out[pos]; //change these ids too
			       img_out[above]=img_out[pos];
			       if(pos1!=pos2) peakcount--; /* may have merged two droplets */
			       //might be able to handle some cases specially
			       //and thus help avoid "complicated" loop at end.
			       //For example track back along a list of 2's, if
			       //it ends with a 1 could merge now.
			       }
			   break;
			   }
		    }
	       oldtyp=typ;
	       above++; curr++; left++;
	       }
	    }
	npeak[0]=peakcount;

	if(complicated) { /* may still need to merge middles of some droplets. */
	    for (curr=0;curr<ncol*nrow;curr++) {
		if(img_out[curr]){
		    //find ultimate id
		    pos=img_out[curr]-1;
		    while(img_out[pos]!=pos+1) pos=img_out[pos]-1;
		    img_out[curr]=img_out[pos];
		    }
		}
	    }
}

extern void raw_photonize(long int *img_out, long int *img_in, long int *bimg, long int *npix, long int *adus, long int *idlist, int npeak, int MINADU, int ADUPPHOT);

void raw_photonize(long int *img_out, long int *img_in, long int *bimg, long int *npix, long int *adus, long int *idlist, int npeak, int MINADU, int ADUPPHOT) {
/* Convert droplets to photons. Uses simple rule to place photon at
 * largest pixel in droplet, reduces it and repeats for number of photons
 * in droplet.
 */

	int i,j,id,nophots,pos,pmax,max;
	/* for each droplet */
	for(i=0;i<npeak;i++) {
	    id=idlist[i]-1;
	    //printf("id=%d\n",id+1);
	    nophots=0;
	    if(adus[i]>MINADU) nophots = 1+(adus[i]-MINADU)/ADUPPHOT;
	    for(j=0;j<nophots;j++) {
	       //find max in droplet and delete for each photon
	       pos=id;
	       max=-100000000; // negative infinity
	       pmax=pos;
	       do {
		  //printf("pos=%d\n",pos+1);
	  //Are we building in correlations by using first max?
	  //if(img_in[pos]==max) printf("double max %d,np=%d\n",pos+1,nophots);
	          if(img_in[pos]>max) {
	             pmax=pos;
	             max=img_in[pos];
	             }
		  pos=bimg[pos]-1;
	          }
	       while(pos!=id);
	       //printf("pmax=%d\n",pmax+1);
	       img_out[pmax]++;
	       img_in[pmax] -= ADUPPHOT;
	       }
	    }
}
