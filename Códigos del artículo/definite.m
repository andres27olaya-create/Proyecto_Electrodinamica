function [value] = definite(prim,x1,x2,y1,y2) 
% "definite" is just the combination of four values of the 2D-primitive  
% "prim" of a function f(x,y). 
% Therefore, it gives the value of the double integral of f(x,y) 
% over the rectangular domain (x1,x2,y1,y2) 
value=prim(x1,y1)+prim(x2,y2)-prim(x1,y2)-prim(x2,y1); 
end