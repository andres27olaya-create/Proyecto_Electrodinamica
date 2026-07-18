% StatMoM computes the MoM matrix element corresponding to the interaction  
% between two parallel xy-cells with identical sizes&orientations and  
% centers in (xb,yb,tb) and (xt,yt,zt). 
% If the distance between the cells' centers is greater than a threshold  
% value given by tresh*max([a b]), then the interaction is approximated by 
% the corresponding GF value. Otherwise, Galerkin (AvDiscrGF) values are 
% % used if flagG=1 and Point-Matching (DiscrGF) values if flagG=0.  
 
% J.R. Mosig,EPFL, June 2023 
 
function result = StatMoM(xb,yb,zb,xt,yt,zt,a,b,tresh,flagG) 
xc=abs(xb-xt); 
yc=abs(yb-yt); 
zc=abs(zb-zt); 
r=sqrt(xc^2+yc^2+zc^2);      % distance between cells' centers 
rh=sqrt(xc^2+yc^2);          % horizontal distance between cells' centers 
if(rh>tresh*max([a b]))      % a possible criterion to select MoM strategies 
result=1/r;               
else 
if(flagG==1)  
result=AvDiscrGF(xc,yc,zc,a,b);  % Galerkin values 
else 
result=DiscrGF(xc,yc,zc,a,b);    
end 
end 