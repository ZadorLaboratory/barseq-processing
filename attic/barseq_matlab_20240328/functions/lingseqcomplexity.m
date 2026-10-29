function u=lingseqcomplexity(seq)
%calculate the linguistic sequece complexity of the given DNA sequece of n x 1. output u is
%the vocabulary usage measure of seq (Trifonov 1990)

u=ones(length(seq),1);

for i=1:length(seq)
    subseq=cell(length(seq)-i+1,1);
    for n=1:length(seq)-i+1
        subseq{n}=upper(seq(n:n+i-1));
    end
    c=length(unique(subseq));
    w=min(4^i,length(seq)-i+1);
    u(i)=c/w;
end