��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.22v2.9.1-132-g18960c44ad38̨
�
Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_17/kernel/v
�
*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_16/kernel/v
�
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes

:@@*
dtype0
�
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:@*
dtype0
�
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/dense_15/kernel/v
�
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes
:	�@*
dtype0
�
Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_17/kernel/m
�
*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_16/kernel/m
�
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes

:@@*
dtype0
�
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:@*
dtype0
�
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/dense_15/kernel/m
�
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes
:	�@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:@*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:@*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:@@*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:@*
dtype0
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	�@*
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
i
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
variance
b
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes	
:�*
dtype0
a
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namemean
Z
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes	
:�*
dtype0
�
ConstConst*
_output_shapes
:	�*
dtype0*�
value�B�	�"�v�A��:B�$>B��>BZ.B(�XB0@B��NB
�VBvcYB��WBN�\B�^B�_[B��OBJ'[BFmB�eB�>B�MB�FSB*SBf�vBJaB��fB��nB�KyB�wiB�`bB�oB84oB��KB��KB��dBHrBLWuB�)lB��~BM~B�|�BP�BF�qB��B�)�B�]�B��B�܃B���B�ƋB���B$�B���Bn�B��B^��B*
�B_�Bn��B~�B0��Bc�B7΁B{iBc��B�a�B�&�B�܂Bg}B���By�B:X�B7�BDϚBf�B`1�B���B<��B�<�B��B�\�B���BhΝB��BLS�B ��B�`yBNs�B�zB�ޕB���B*��B{��Br��B|x�B�٧B��BN1�B&��B��B�O�B��Bh�B�B��B��Bi,�B�B��BPʰB�C�B5�B��B�R�B�Q�B���B(ٻB.�B���B�ѭB�B`��BH��BtI�B��Bb�B C�B�̸BC��B���B��B�̾B��B�}�B��B���B�l�B���B���BN(�BU��B�K�B�}�BJ]�B6H�B>�B���B���BtG�BY�Bz�BZ��B���B��B���B��B���B|�B3��B�7�B�u�BR�B�p�B��B �B���B���BJ��B��B�<�B�5�BX�B�p�B���B��B|��B�q�Bb��B���B�d�Bf�B�W�B�m�B
a�B���B�	�B0�B�P�B���B�5�B���B�B��B��B���B�;�B�q�B�W�B���B8�B���Bm�B���B~%�B���B���B�{�B��B>��B��Bi�BH/�B�V�B�Bi��B֔�B W�B�{�Be��B�l�B�!�B��B]�B.��BO�B��B_2�B���B��B�a�B���Bvb�B"��B&�BS{�B���BƩ�B%L�B���B�-�B�B\0�Bf5�B���B��B��BO�B2e�B"��B�V�BF�B�
�B�P�B��B'��B��B%��B��B.��BH��B&�B2�Br�Bj��B	�B���Bo��B	�Bԕ�B"� C��BZv�BU��B��B��Bh)CIG C0��B{��B [�B�#�B6��B,��B�tC3�C���B�}C$��BQ��BT��B'�CJ7C
]�B8$�Bx��B�]�B���B�Ch+C�%�B�4�B6��B��Bx� C�C��Cvv C�m C� CZ� C7�Cu� C�X�B)�C�V C���B��Bͪ�B���B���B���B��B���B%%C}-C�C�FC�Ct	 C�� C�z Cjt�B�	�B��B���B`�B�C�%CZCM�C�Z	C0� C��Cz�	C�C	C�Cx�C�@C��C4XC��C2C�}C aCa�CB�C��C;�C�C"XC�BCEHCTC��C�mC9��B�$C-C�fC�,C�C�lC��C��C�lC{oC)C`C�bCo�
Cr�
C>�C.�C��
C�<Cb<C�CS�C�qC��C�'C<�C*�C�Ct5
C�ZC�GCσC\5C"�C��	Ch�	C.C��CNrCJ�C�C��C��C�CeC�KC˺CwCC̕C�CCe
C��C�C+C�(C<n	C�C=�
CZ�C��C�Cj�CI$	C�jC�kC�QC��C�C�9	C�=C��CCC�CƏC&KCZC�C� C&�C�/C�(C�;	CC�C��C?�C�C6CI�C
�	C�CCIV	C�XC9�Co�	C9 C��	C^�C��CD�CdQC��	C�CYCV�CdYCh��B3'C5�
C��C~
CpI	C�lCҶ
C� 
C�nC�y
CR�
C��C�C��C��
CfC��C4	C͕C�CʓC�C2�
CyC$MCڸC��C&C��Ce&C&�
Cu
CfC�y	CT�
C�
C��C+�C��C	+
C�CWUC�]C,�CZ�C��CU{C��	C��CTC��C��C��Cz+C7�C~]C`�
CbzCj�C�
C�r
C*�C"C��CpCCj�
C,�CT�Cl�C��C7TC0|C��
C�Cq(C��C��C�Co�	C��C�'C�s	C�'	C�8Cv�
C@0C֍C�C�C��C��
Cm�CEwC0�C��C{�C�C�bC��CM�
C��C*�CRlC
}C��
C�C!NC��C�YCњC�C�C�yCٯC\eCCC��C��CC��C؛C�Cr�CqC�#C]�C�C�C�C��C��C��	C<�C٨Cc�C��C�C��C�C,7C��Cz�C��Cg�C��Cf�C�Ch)CtYC��CN�Cx�C�C��C�%C
sC�rC��C�C��C�
C��C�C��CQCȫC��C�DCYbCS�C�'C��CACCY�Ck1	C*(C�C�~C�cC��C2CC�C�]Cv�C�_C�xCF�C��CE�CZDC�C�uC8�C�CC<�CC/�C��C��C%�C�C�CR�C^yC�bCC��C^�C��C^HC�XCzrCp�C��C�C��Cg�C�nC&�C��C�pC։C��C*Cd�C��C� C�:C��C_GC�C�dC�C$C�DCDPC��C�<C�gCf�C��C�C:5C�JC�3C
�Cj�C^mC��Cr�C8�C�/C��C��C�FC��C�C"�CPC�_CP�CfC�CxC�2Cf�C�gCj�C��Cp#C��Cg�C�:C-�C$�C��C2�C
^C+C4\C�3C�OC�mC^QC�C��C�2C]�C��C=�C��C?C��C��CСCr1C�� C�C^�C��Cr�C�=CǻC��#C��C.�CJCX�C$�C
?C�C�#C[	C%�Cf$C��$ClHC��C&C:]C8`C�0C��CI�C� C�C �CؐC��C��CXC(� C��$C�0C�CnCƝC�ECAC��C��C�C�!C~�C\�CkC��CeC4 C�wCne C6\%CrO#C�b%CNj$C�HC�>C�Cn,$CL�C�CT�C&C� C�,C��#C�C��C.�C�C$�C~�"C�&C��Czp!C�!C�( C��CT�!C��$C�"C Ce!C.�C3CمC�� C��!C��#CX;"CB�#ChC� C�qC��C�C
{ C�CFC\0C3I"C#� C�!Cé%C��%C �C�!C �!Ch� C�t"C�-$C  C@� C�%Cʠ C��CIC��%C�#C�%C
� C�V&C�7C��"C^� CF�C��C^-Cm|C�C��C� $C�: C�C��Cc�"Cw�"C� C�H!Cy Cl(C��Czn#C$C�CmmC�]C�C��C�"Cv%C��C8�"C�	"CF�C�TC�C�C|cC��C��C��C��C�� C��!C��!C��C�%CR�#C�CK�$C7!C� C�� CzC`6"C�"Ch�&Cփ%C��2C�6CHu4C�4C�7C��5C�>:C�O6C�>5C �9C*�6C$3Cm�8CT�:C��7CF�9Cʠ8C�z8Cd�8C�;:CD�/CR�3CP)7C�o8C�7C2�6C�9C�8C�/C(>9C8C0�:Cf{8CPk8Ch�8C!O3C8�7Cf<C��9C�o6C�9C<�@C~�;C��;C��@C�/<C
�
Const_1Const*
_output_shapes
:	�*
dtype0*�
value�B�	�"�N�El(G$oG�6GK��Fg)G�=�F@G�>GεGP�G��GCG�G�G�G�)G�$G��Fq�G�Gf5Go�0G�]Gv�G� G��-G��G��G$G-� G�G��G��GJtG��"GZ��F}i*G�RGF�7GVC#G<�F�A"G� *G�s.G �6GJX%G��$G��3G �3Gp>GJG�AG��&Gz-G58)GMP/GұGu�3G�@;Gy�1G�l�F���FQ0+G��F��>G%��F>��F��8G�;GBu@G�y8G6�JG�~GG��BGGa8G�FGc*7G9�7G�9GKG��?GI?GnFG��AG�2�F?�4G�N�F�H4GF�2G��7G��5G��9G�bEG��CGNA>G׮HGZxJG}tKG�>G;5<G̳HGDPG�FG4I<G,EG�)HG-�HG��CG`�DG�
GG��FGLBG�dBG��LG�UG���F\�DGW�G��PGj�PG�iJG�^OG��KGcFG�^HG�;IG�IG�GGF9G�NGG`�PG�LG4�PG� PGl�LGx�QGx�RG~�FJ�?G��fG�Y,G��LG?MGr�SG�4G'h=GR�SG=�\G�QG1�RG�VZG(6PG�ASG �%G�3XGB�TG�VGܺPGpVG��RGBIUG�lSG��VG�
G*�VG��ZG6�SG�=G��[G
5SG�^Gp�iG��eG�`G�mvGLCkG�_dG&�eG�}sGt�uG�\G�M`G�	fG�*Gf]iG�tGۂtG��pG��GbEzG��mG��fG+0eG��tGb9GRkyG*�xG�hG��fG��pGj�^G~meG��nG�xG��WG�,cG �GG2�xG�3wG�.eG��oG\�QG�hG�pG�L~G��kG�4[G�dGd#qG�cGCGPFG	�vGj�aG¡hG��qG��mG�	dG�meG�RGޝjG�]>G�}cGPoG@�:G�qG�LG<qGR�jG�MwG>�G�|GKrG��xG��xG>�G��CG�{xG�xG�{xGT�GT�pGi�G�G�uGg� G�$GG��G�mG�I�G�{&GRrvG�kG�rG��tG�2G�0{GH��G�ZwG�G�6G��sG|�~G���G�tG��GGL}}GeGyGT�@G���GPe�G3BrGT�G��QGh|G uyGiX�G�H�G��wG�{GCSpG
��G"�zG���G�7�GؼvG ��G�m~G�#�G�A�G��G��G	�G�_�G�ځG��G�σGgc�Gn-tGb �G�5�G�8{G#&~G�A{Gv�yGȞyG���G��uG&�9G�B�G^��G�C�Gv�|Gt~Gh�}GV�Gx�G�V�G�zG�&GT�JG��eGJƅGRŇGyr�Gh��G[�G��yG؟G��G���G�P�G�τGF�G�0�G�YcG�؄G�s�G�3�GƇG�B�G�gGz�GLy�Gݞ}G��G�ąG�x�G<��G�'�G���G�.=G�4�G�QG�)�G�T�G�oG��G[��G+��G�w�G�'�G�/�Gb_bG=~GֆGT(�G �G��GZ��G���G�y�GЉ�G�ˁG{&�G���G���GF �G�WaG.X�G�{�G��cG:��GRi�G�wG�?jG�luG�"wG(`GzJaGfXG��HG��`G|IG?�`G^rUG��bG�&WG^�XG&,XG�#ZG~�bGx�aGf�jG��_G�D_GLDQGn�lG�bG��ZG�hjG>roG͐mG	aG��iG��hG�NGC^G`�gG��]G`vVGaikG�?TG��dG8�RG��fG�*hG��bGZ�dG�goG�AZG��bG$leG�maG�iGtbxG�MyG�m\GD�cG&cG�YGG^\G�IcG�WGR�`G�`G�RGb�\GhfGt?]G�_G>�EG�ThGjJG�QSG��JG�GG6VgGVxZG�67GbkG@�SG�_G��fGt7hG�G\Gf�dG9\G�>NG�XG^WG�NfG�jnG�MGrVGA^G�[G,�gG(�IG�9sG1�NG�}kG?�YGvEfG��LG�O\G�URG�gG�KGf,OG�n^G9�fG�vVG�VG�BYGI�UG�VYG�AG�UG�zQG�wG6�UG��\G�_aG(�kGǑNG�;_G�:RG�&nGloPGd�fG�RQG��aG�KG�R^G��NG�\GEKG��lG��WG��[G�enG��VG\�LG�=_GyTG�^JG*�SG=�TG��RG��PG~iiGF�YG�BLG�3FGr�cGp4ZG�|fG��UG:�RG>�YG��UG"TGG�oG>6YG�7WG^diG�PoGb:kG�:[G�VG�
^G��oG��qGj�sG2^WGH6rG�bGL>qG�:ZG�oG!TG��RG;SG~�UG��WG��rG��XG�uG��UGguGLYG�tG�^G�u^G�8aG$�aGj\G6XhGS�dG�sG�$oG��pG�bG��]GL�_G�]G"<^G��oG�rG�^G�1_Gt�\G^vG��]G^�`G�	cG�ZG��]Gx�rG�!WGj�`G2eeGu^zG��lG��YG��dG�uG��{GLG�G �pG2�dG�dG]ulGhG�UbG��aG�eG`�`G��cG�`pG�XG�YG�hG��rGmG�eGfG��oG��fG�jGJ�eG�.yGz4gG�S`GLH`GP%QG��qGb�yG;vGF�uG"rG|�bGV�gG��fG�ˁGB�kGu�qGϥhGB�wGN�_G��PG�taG��eG�H^GPaG~�gG�#cG�5oG�E_G�|yG4�jG�eG�rG��cG��bG`\G�6kGn�jG�nG޶qG|�dG�XlG�ˊG%�Gנ�GJ:kG�vG�shGx�kG;)�G
�GS�rGJXiG���G�;pG��}G�zjGH�cG`�bG�kG`;�G$B|Gl�G��G{mcG���GquG�7}GJ�wG�zG4A{Gx�fGLXgGmcG�HiG�RrG�6lGf{G���G�NpG�GL�mG�5�G�FsG�lG�VwG��qG�cG2EyG�&|G��wG��tGR��GRxG�iG�L�G%yG0}G�wGT(�G��{G�9xG��{G�G�%wG��|G[�GD�G��zG߬iG^wG
~GƬyG]ևG$�G�g�G@�G��GꯄG�izGp�Gz��G^��G2�G���G��G:�G!��GvL�G�J�G��GlʌG,6�G3�|GO��G�y�Gظ�GZ�pG�s�Gs<vG���G!A�Gz��G�G<#�GZ�G|+�Gn��G��rGn�G规G�zwGwl�GDmG�AyGz�Gf�G�ݍG�m�G��G�G���G(�G�՗G�E�G�5�Gi�G� �G8�G���Gz�GdG�GY�G���G!��G���GPG�Gԧ�G
�Go��G_��G��G$��G4��G�S�G��G��G"��G���GpߓG�.�G��GMp�G�2�G�#�G��G���G��G��G߈G7��G���G2o�G�5�G��GC�G}q�G֧�G��G.��G�'�Gh�G�@�Gg��G
؂G��G�bG��GQ��G�|�G�G���GG!�G�G7�G�Z�G�ÂG�b�G\�G��G���G�t�G���G��G V�G�;�G��G灑G�̊GP��G��G��G�ۍGPȑG^ēG���Gr�G]��G���GA�G���G�k�G��G$�GR(�G�C�Gu��G�3�G߅G�q�G���G���Gyg�G�G��G���G�B�G�܎G��G�G�G�w�G��G67�G�4�G�G�{�GV��Gj��G�ֈG'#�G;�GL1�G.N�G���Gr�G���G}�G�d�G��G�&�Gb=�G�ȆG�8�G2ǊG�=�G���G�@�G%ēG�l�GA�G���G���GЫ�G�=�GB��G �G ܊G�GDJ�G�˭G�o�G~z�G��GR��G�F�GUH�G���G��G+Z�G�'�G$6�G���G���G��G���G��Gyu�G2��G���GS+�G���GOY�Gh��G��Gl�G���G.��G�<�Gy�Gѯ�G�
�G�N�GT<�G�M�G���G}8�G>��GZ��G��G}/�G��G&$�G���GJ�G��G

NoOpNoOp
�5
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�5
value�5B�5 B�5
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator* 
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
C
0
1
2
3
4
-5
.6
57
68*
.
0
1
-2
.3
54
65*
* 
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
<trace_0
=trace_1
>trace_2
?trace_3* 
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
* 
�
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratemnmo-mp.mq5mr6msvtvu-vv.vw5vx6vy*

Iserving_default* 
* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_15layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*

Jtrace_0* 

0
1*

0
1*
* 
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ptrace_0* 

Qtrace_0* 
_Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_15/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

Wtrace_0
Xtrace_1* 

Ytrace_0
Ztrace_1* 
* 

-0
.1*

-0
.1*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

`trace_0* 

atrace_0* 
_Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*

50
61*
* 
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
_Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_17/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*
'
0
1
2
3
4*

i0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
j	variables
k	keras_api
	ltotal
	mcount*

l0
m1*

j	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_16/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_17/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_17/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_16/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_17/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_17/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
%serving_default_normalization_5_inputPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
StatefulPartitionedCallStatefulPartitionedCall%serving_default_normalization_5_inputConstConst_1dense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_161625
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount_1/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOpConst_2*)
Tin"
 2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_161978
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_15/kernel/mAdam/dense_15/bias/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/dense_17/kernel/mAdam/dense_17/bias/mAdam/dense_15/kernel/vAdam/dense_15/bias/vAdam/dense_16/kernel/vAdam/dense_16/bias/vAdam/dense_17/kernel/vAdam/dense_17/bias/v*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_162072��
�

�
D__inference_dense_16_layer_call_and_return_conditional_losses_161357

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_dense_17_layer_call_fn_161859

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_161373o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�*
�
!__inference__wrapped_model_161308
normalization_5_input&
"sequential_5_normalization_5_sub_y'
#sequential_5_normalization_5_sqrt_xG
4sequential_5_dense_15_matmul_readvariableop_resource:	�@C
5sequential_5_dense_15_biasadd_readvariableop_resource:@F
4sequential_5_dense_16_matmul_readvariableop_resource:@@C
5sequential_5_dense_16_biasadd_readvariableop_resource:@F
4sequential_5_dense_17_matmul_readvariableop_resource:@C
5sequential_5_dense_17_biasadd_readvariableop_resource:
identity��,sequential_5/dense_15/BiasAdd/ReadVariableOp�+sequential_5/dense_15/MatMul/ReadVariableOp�,sequential_5/dense_16/BiasAdd/ReadVariableOp�+sequential_5/dense_16/MatMul/ReadVariableOp�,sequential_5/dense_17/BiasAdd/ReadVariableOp�+sequential_5/dense_17/MatMul/ReadVariableOp�
 sequential_5/normalization_5/subSubnormalization_5_input"sequential_5_normalization_5_sub_y*
T0*(
_output_shapes
:����������x
!sequential_5/normalization_5/SqrtSqrt#sequential_5_normalization_5_sqrt_x*
T0*
_output_shapes
:	�k
&sequential_5/normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
$sequential_5/normalization_5/MaximumMaximum%sequential_5/normalization_5/Sqrt:y:0/sequential_5/normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	��
$sequential_5/normalization_5/truedivRealDiv$sequential_5/normalization_5/sub:z:0(sequential_5/normalization_5/Maximum:z:0*
T0*(
_output_shapes
:�����������
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_5/dense_15/MatMulMatMul(sequential_5/normalization_5/truediv:z:03sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
sequential_5/dropout_5/IdentityIdentity(sequential_5/dense_15/Relu:activations:0*
T0*'
_output_shapes
:���������@�
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
sequential_5/dense_16/MatMulMatMul(sequential_5/dropout_5/Identity:output:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+sequential_5/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
sequential_5/dense_17/MatMulMatMul(sequential_5/dense_16/Relu:activations:03sequential_5/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_5/dense_17/BiasAddBiasAdd&sequential_5/dense_17/MatMul:product:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
IdentityIdentity&sequential_5/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp,^sequential_5/dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_17/MatMul/ReadVariableOp+sequential_5/dense_17/MatMul/ReadVariableOp:g c
0
_output_shapes
:������������������
/
_user_specified_namenormalization_5_input:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�

�
-__inference_sequential_5_layer_call_fn_161712

inputs
unknown
	unknown_0
	unknown_1:	�@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_161502o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�
c
*__inference_dropout_5_layer_call_fn_161813

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_161439o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
$__inference_signature_wrapper_161625
normalization_5_input
unknown
	unknown_0
	unknown_1:	�@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_161308o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
0
_output_shapes
:������������������
/
_user_specified_namenormalization_5_input:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�"
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161744

inputs
normalization_5_sub_y
normalization_5_sqrt_x:
'dense_15_matmul_readvariableop_resource:	�@6
(dense_15_biasadd_readvariableop_resource:@9
'dense_16_matmul_readvariableop_resource:@@6
(dense_16_biasadd_readvariableop_resource:@9
'dense_17_matmul_readvariableop_resource:@6
(dense_17_biasadd_readvariableop_resource:
identity��dense_15/BiasAdd/ReadVariableOp�dense_15/MatMul/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOpl
normalization_5/subSubinputsnormalization_5_sub_y*
T0*(
_output_shapes
:����������^
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:	�^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	��
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*(
_output_shapes
:�����������
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_15/MatMulMatMulnormalization_5/truediv:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:���������@m
dropout_5/IdentityIdentitydense_15/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_16/MatMulMatMuldropout_5/Identity:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�'
�
__inference_adapt_step_161670
iterator%
add_readvariableop_resource:	 &
readvariableop_resource:	�(
readvariableop_2_resource:	���AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*(
_output_shapes
:����������*'
output_shapes
:����������*
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 a
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0Q
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes	
:�Y
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes	
:�H
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes	
:�e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0W
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes	
:�J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @K
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes	
:�g
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes	
:�*
dtype0W
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes	
:�F
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes	
:�W
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes	
:�L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @O
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes	
:�[
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes	
:�J
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes	
:�J
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes	
:��
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator
�

�
D__inference_dense_15_layer_call_and_return_conditional_losses_161803

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161502

inputs
normalization_5_sub_y
normalization_5_sqrt_x"
dense_15_161485:	�@
dense_15_161487:@!
dense_16_161491:@@
dense_16_161493:@!
dense_17_161496:@
dense_17_161498:
identity�� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCalll
normalization_5/subSubinputsnormalization_5_sub_y*
T0*(
_output_shapes
:����������^
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:	�^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	��
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*(
_output_shapes
:�����������
 dense_15/StatefulPartitionedCallStatefulPartitionedCallnormalization_5/truediv:z:0dense_15_161485dense_15_161487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_161333�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_161439�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_16_161491dense_16_161493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_161357�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_161496dense_17_161498*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_161373x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�*
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161783

inputs
normalization_5_sub_y
normalization_5_sqrt_x:
'dense_15_matmul_readvariableop_resource:	�@6
(dense_15_biasadd_readvariableop_resource:@9
'dense_16_matmul_readvariableop_resource:@@6
(dense_16_biasadd_readvariableop_resource:@9
'dense_17_matmul_readvariableop_resource:@6
(dense_17_biasadd_readvariableop_resource:
identity��dense_15/BiasAdd/ReadVariableOp�dense_15/MatMul/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOpl
normalization_5/subSubinputsnormalization_5_sub_y*
T0*(
_output_shapes
:����������^
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:	�^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	��
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*(
_output_shapes
:�����������
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_15/MatMulMatMulnormalization_5/truediv:z:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:���������@\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout_5/dropout/MulMuldense_15/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:���������@b
dropout_5/dropout/ShapeShapedense_15/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@�
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@�
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_16/MatMulMatMuldropout_5/dropout/Mul_1:z:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�

�
-__inference_sequential_5_layer_call_fn_161691

inputs
unknown
	unknown_0
	unknown_1:	�@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_161380o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�p
�
"__inference__traced_restore_162072
file_prefix$
assignvariableop_mean:	�*
assignvariableop_1_variance:	�$
assignvariableop_2_count_1:	 5
"assignvariableop_3_dense_15_kernel:	�@.
 assignvariableop_4_dense_15_bias:@4
"assignvariableop_5_dense_16_kernel:@@.
 assignvariableop_6_dense_16_bias:@4
"assignvariableop_7_dense_17_kernel:@.
 assignvariableop_8_dense_17_bias:&
assignvariableop_9_adam_iter:	 )
assignvariableop_10_adam_beta_1: )
assignvariableop_11_adam_beta_2: (
assignvariableop_12_adam_decay: 0
&assignvariableop_13_adam_learning_rate: #
assignvariableop_14_total: #
assignvariableop_15_count: =
*assignvariableop_16_adam_dense_15_kernel_m:	�@6
(assignvariableop_17_adam_dense_15_bias_m:@<
*assignvariableop_18_adam_dense_16_kernel_m:@@6
(assignvariableop_19_adam_dense_16_bias_m:@<
*assignvariableop_20_adam_dense_17_kernel_m:@6
(assignvariableop_21_adam_dense_17_bias_m:=
*assignvariableop_22_adam_dense_15_kernel_v:	�@6
(assignvariableop_23_adam_dense_15_bias_v:@<
*assignvariableop_24_adam_dense_16_kernel_v:@@6
(assignvariableop_25_adam_dense_16_bias_v:@<
*assignvariableop_26_adam_dense_17_kernel_v:@6
(assignvariableop_27_adam_dense_17_bias_v:
identity_29��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_count_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_15_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_15_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_16_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_16_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_17_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_17_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_15_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_15_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_16_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_16_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_17_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_17_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_15_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_15_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_16_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_16_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_17_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_17_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
D__inference_dense_16_layer_call_and_return_conditional_losses_161850

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_161344

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_161439

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
-__inference_sequential_5_layer_call_fn_161399
normalization_5_input
unknown
	unknown_0
	unknown_1:	�@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_161380o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
0
_output_shapes
:������������������
/
_user_specified_namenormalization_5_input:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�

�
D__inference_dense_15_layer_call_and_return_conditional_losses_161333

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161596
normalization_5_input
normalization_5_sub_y
normalization_5_sqrt_x"
dense_15_161579:	�@
dense_15_161581:@!
dense_16_161585:@@
dense_16_161587:@!
dense_17_161590:@
dense_17_161592:
identity�� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall{
normalization_5/subSubnormalization_5_inputnormalization_5_sub_y*
T0*(
_output_shapes
:����������^
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:	�^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	��
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*(
_output_shapes
:�����������
 dense_15/StatefulPartitionedCallStatefulPartitionedCallnormalization_5/truediv:z:0dense_15_161579dense_15_161581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_161333�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_161439�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_16_161585dense_16_161587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_161357�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_161590dense_17_161592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_161373x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:g c
0
_output_shapes
:������������������
/
_user_specified_namenormalization_5_input:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�

�
-__inference_sequential_5_layer_call_fn_161542
normalization_5_input
unknown
	unknown_0
	unknown_1:	�@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_161502o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
0
_output_shapes
:������������������
/
_user_specified_namenormalization_5_input:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�
F
*__inference_dropout_5_layer_call_fn_161808

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_161344`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_161818

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�=
�
__inference__traced_save_161978
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop&
"savev2_count_1_read_readvariableop	.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableop
savev2_const_2

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop"savev2_count_1_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *+
dtypes!
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�: :	�@:@:@@:@:@:: : : : : : : :	�@:@:@@:@:@::	�@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:�:!

_output_shapes	
:�:

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 	

_output_shapes
::


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161380

inputs
normalization_5_sub_y
normalization_5_sqrt_x"
dense_15_161334:	�@
dense_15_161336:@!
dense_16_161358:@@
dense_16_161360:@!
dense_17_161374:@
dense_17_161376:
identity�� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCalll
normalization_5/subSubinputsnormalization_5_sub_y*
T0*(
_output_shapes
:����������^
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:	�^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	��
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*(
_output_shapes
:�����������
 dense_15/StatefulPartitionedCallStatefulPartitionedCallnormalization_5/truediv:z:0dense_15_161334dense_15_161336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_161333�
dropout_5/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_161344�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_16_161358dense_16_161360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_161357�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_161374dense_17_161376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_161373x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�
�
)__inference_dense_15_layer_call_fn_161792

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_161333o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
D__inference_dense_17_layer_call_and_return_conditional_losses_161373

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161569
normalization_5_input
normalization_5_sub_y
normalization_5_sqrt_x"
dense_15_161552:	�@
dense_15_161554:@!
dense_16_161558:@@
dense_16_161560:@!
dense_17_161563:@
dense_17_161565:
identity�� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall{
normalization_5/subSubnormalization_5_inputnormalization_5_sub_y*
T0*(
_output_shapes
:����������^
normalization_5/SqrtSqrtnormalization_5_sqrt_x*
T0*
_output_shapes
:	�^
normalization_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization_5/MaximumMaximumnormalization_5/Sqrt:y:0"normalization_5/Maximum/y:output:0*
T0*
_output_shapes
:	��
normalization_5/truedivRealDivnormalization_5/sub:z:0normalization_5/Maximum:z:0*
T0*(
_output_shapes
:�����������
 dense_15/StatefulPartitionedCallStatefulPartitionedCallnormalization_5/truediv:z:0dense_15_161552dense_15_161554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_161333�
dropout_5/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_161344�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_16_161558dense_16_161560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_161357�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_161563dense_17_161565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_161373x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:������������������:	�:	�: : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:g c
0
_output_shapes
:������������������
/
_user_specified_namenormalization_5_input:%!

_output_shapes
:	�:%!

_output_shapes
:	�
�	
d
E__inference_dropout_5_layer_call_and_return_conditional_losses_161830

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_dense_16_layer_call_fn_161839

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_161357o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
D__inference_dense_17_layer_call_and_return_conditional_losses_161869

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
`
normalization_5_inputG
'serving_default_normalization_5_input:0������������������<
dense_170
StatefulPartitionedCall:0���������tensorflow/serving/predict:ď
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
_
0
1
2
3
4
-5
.6
57
68"
trackable_list_wrapper
J
0
1
-2
.3
54
65"
trackable_list_wrapper
 "
trackable_list_wrapper
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
<trace_0
=trace_1
>trace_2
?trace_32�
-__inference_sequential_5_layer_call_fn_161399
-__inference_sequential_5_layer_call_fn_161691
-__inference_sequential_5_layer_call_fn_161712
-__inference_sequential_5_layer_call_fn_161542�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z<trace_0z=trace_1z>trace_2z?trace_3
�
@trace_0
Atrace_1
Btrace_2
Ctrace_32�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161744
H__inference_sequential_5_layer_call_and_return_conditional_losses_161783
H__inference_sequential_5_layer_call_and_return_conditional_losses_161569
H__inference_sequential_5_layer_call_and_return_conditional_losses_161596�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
�B�
!__inference__wrapped_model_161308normalization_5_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
Diter

Ebeta_1

Fbeta_2
	Gdecay
Hlearning_ratemnmo-mp.mq5mr6msvtvu-vv.vw5vx6vy"
	optimizer
,
Iserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:�2mean
:�2variance
:	 2count
�
Jtrace_02�
__inference_adapt_step_161670�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ptrace_02�
)__inference_dense_15_layer_call_fn_161792�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zPtrace_0
�
Qtrace_02�
D__inference_dense_15_layer_call_and_return_conditional_losses_161803�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQtrace_0
": 	�@2dense_15/kernel
:@2dense_15/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
Wtrace_0
Xtrace_12�
*__inference_dropout_5_layer_call_fn_161808
*__inference_dropout_5_layer_call_fn_161813�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zWtrace_0zXtrace_1
�
Ytrace_0
Ztrace_12�
E__inference_dropout_5_layer_call_and_return_conditional_losses_161818
E__inference_dropout_5_layer_call_and_return_conditional_losses_161830�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zYtrace_0zZtrace_1
"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
`trace_02�
)__inference_dense_16_layer_call_fn_161839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
�
atrace_02�
D__inference_dense_16_layer_call_and_return_conditional_losses_161850�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
!:@@2dense_16/kernel
:@2dense_16/bias
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
gtrace_02�
)__inference_dense_17_layer_call_fn_161859�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zgtrace_0
�
htrace_02�
D__inference_dense_17_layer_call_and_return_conditional_losses_161869�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
!:@2dense_17/kernel
:2dense_17/bias
5
0
1
2"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_5_layer_call_fn_161399normalization_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_5_layer_call_fn_161691inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_5_layer_call_fn_161712inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_5_layer_call_fn_161542normalization_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161744inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161783inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161569normalization_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161596normalization_5_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
$__inference_signature_wrapper_161625normalization_5_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_adapt_step_161670iterator"�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_15_layer_call_fn_161792inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_15_layer_call_and_return_conditional_losses_161803inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_5_layer_call_fn_161808inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
*__inference_dropout_5_layer_call_fn_161813inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
E__inference_dropout_5_layer_call_and_return_conditional_losses_161818inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
E__inference_dropout_5_layer_call_and_return_conditional_losses_161830inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_16_layer_call_fn_161839inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_16_layer_call_and_return_conditional_losses_161850inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_17_layer_call_fn_161859inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_17_layer_call_and_return_conditional_losses_161869inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
j	variables
k	keras_api
	ltotal
	mcount"
_tf_keras_metric
.
l0
m1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
':%	�@2Adam/dense_15/kernel/m
 :@2Adam/dense_15/bias/m
&:$@@2Adam/dense_16/kernel/m
 :@2Adam/dense_16/bias/m
&:$@2Adam/dense_17/kernel/m
 :2Adam/dense_17/bias/m
':%	�@2Adam/dense_15/kernel/v
 :@2Adam/dense_15/bias/v
&:$@@2Adam/dense_16/kernel/v
 :@2Adam/dense_16/bias/v
&:$@2Adam/dense_17/kernel/v
 :2Adam/dense_17/bias/v
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant�
!__inference__wrapped_model_161308�z{-.56G�D
=�:
8�5
normalization_5_input������������������
� "3�0
.
dense_17"�
dense_17���������p
__inference_adapt_step_161670OD�A
:�7
5�2�
�����������IteratorSpec 
� "
 �
D__inference_dense_15_layer_call_and_return_conditional_losses_161803]0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� }
)__inference_dense_15_layer_call_fn_161792P0�-
&�#
!�
inputs����������
� "����������@�
D__inference_dense_16_layer_call_and_return_conditional_losses_161850\-./�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_dense_16_layer_call_fn_161839O-./�,
%�"
 �
inputs���������@
� "����������@�
D__inference_dense_17_layer_call_and_return_conditional_losses_161869\56/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� |
)__inference_dense_17_layer_call_fn_161859O56/�,
%�"
 �
inputs���������@
� "�����������
E__inference_dropout_5_layer_call_and_return_conditional_losses_161818\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
E__inference_dropout_5_layer_call_and_return_conditional_losses_161830\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� }
*__inference_dropout_5_layer_call_fn_161808O3�0
)�&
 �
inputs���������@
p 
� "����������@}
*__inference_dropout_5_layer_call_fn_161813O3�0
)�&
 �
inputs���������@
p
� "����������@�
H__inference_sequential_5_layer_call_and_return_conditional_losses_161569�z{-.56O�L
E�B
8�5
normalization_5_input������������������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_161596�z{-.56O�L
E�B
8�5
normalization_5_input������������������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_161744sz{-.56@�=
6�3
)�&
inputs������������������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_161783sz{-.56@�=
6�3
)�&
inputs������������������
p

 
� "%�"
�
0���������
� �
-__inference_sequential_5_layer_call_fn_161399uz{-.56O�L
E�B
8�5
normalization_5_input������������������
p 

 
� "�����������
-__inference_sequential_5_layer_call_fn_161542uz{-.56O�L
E�B
8�5
normalization_5_input������������������
p

 
� "�����������
-__inference_sequential_5_layer_call_fn_161691fz{-.56@�=
6�3
)�&
inputs������������������
p 

 
� "�����������
-__inference_sequential_5_layer_call_fn_161712fz{-.56@�=
6�3
)�&
inputs������������������
p

 
� "�����������
$__inference_signature_wrapper_161625�z{-.56`�]
� 
V�S
Q
normalization_5_input8�5
normalization_5_input������������������"3�0
.
dense_17"�
dense_17���������