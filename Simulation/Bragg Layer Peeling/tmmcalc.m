% This function calculates the reflection coefficient using coupled mode theory-based trasfer matrix method (CMT-TMM)
function [r,t]=tmmcalc(q,dz,delta)

N=length(q);
Nlambda=length(delta);
MT=zeros(2,2,Nlambda);			% dim T matrix
for m=1:Nlambda					% unit matrix
   MT(:,:,m)=[1 0;0 1];
end;

for j=1:N						% sections
   qabs=abs(q(j));
   for m=1:Nlambda				% freq.
      p=sqrt(qabs^2-delta(m)^2);
      T=[cosh(p*dz)+i*delta(m)/p*sinh(p*dz) q(j)/p*sinh(p*dz) ;conj(q(j))/p*sinh(p*dz) cosh(p*dz)-i*delta(m)/p*sinh(p*dz)];
      MT(:,:,m)=T*MT(:,:,m);	% save accumulated T matrix
   end   
end;

rtemp=-MT(2,1,:)./MT(2,2,:);
ttemp=1./MT(2,2,:);
r=rtemp(:).';								% reflection coefficient
t=ttemp(:).';								% transmission coefficient