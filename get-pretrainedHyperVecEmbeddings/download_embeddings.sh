#!/bin/bash
# downloads hypvec_embeddings from IMS homepage
for id in 1 2 3 4 5 6 7 8 9
	do
	wget http://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/hypvec_embd/hyp_p${id}.gz
	done
cat hyp_p1.gz hyp_p2.gz hyp_p3.gz hyp_p4.gz hyp_p5.gz hyp_p6.gz hyp_p7.gz hyp_p8.gz hyp_p9.gz > hypervec.txt.gz
# rm -f hyp_p*.gz # OPTIONAL  -remove files
# gunzip hypervec.txt.gz # OPTIONAL unzip embeddings to plain text
