��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
7
Square
x"T
y"T"
Ttype:
2	
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.16.12v2.16.0-rc0-18-g5bc9d26649c8͖
�
dense_9/biasVarHandleOp*
_output_shapes
: *

debug_namedense_9/bias/*
dtype0*
shape:2*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:2*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpdense_9/bias*
_class
loc:@Variable*
_output_shapes
:2*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:2*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:2*
dtype0
�
dense_9/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_9/kernel/*
dtype0*
shape:	�2*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	�2*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpdense_9/kernel*
_class
loc:@Variable_1*
_output_shapes
:	�2*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:	�2*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�2*
dtype0
�
dense_8/biasVarHandleOp*
_output_shapes
: *

debug_namedense_8/bias/*
dtype0*
shape:�*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:�*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpdense_8/bias*
_class
loc:@Variable_2*
_output_shapes	
:�*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:�*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
f
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes	
:�*
dtype0
�
dense_8/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_8/kernel/*
dtype0*
shape:���*
shared_namedense_8/kernel
t
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*!
_output_shapes
:���*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpdense_8/kernel*
_class
loc:@Variable_3*!
_output_shapes
:���*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:���*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
l
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*!
_output_shapes
:���*
dtype0
�
conv2d_10/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_10/bias/*
dtype0*
shape:@*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:@*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpconv2d_10/bias*
_class
loc:@Variable_4*
_output_shapes
:@*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:@*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
e
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:@*
dtype0
�
conv2d_10/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_10/kernel/*
dtype0*
shape: @*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: @*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpconv2d_10/kernel*
_class
loc:@Variable_5*&
_output_shapes
: @*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape: @*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
q
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*&
_output_shapes
: @*
dtype0
�
conv2d_9/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_9/bias/*
dtype0*
shape: *
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
: *
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpconv2d_9/bias*
_class
loc:@Variable_6*
_output_shapes
: *
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape: *
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
: *
dtype0
�
conv2d_9/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_9/kernel/*
dtype0*
shape: * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
: *
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpconv2d_9/kernel*
_class
loc:@Variable_7*&
_output_shapes
: *
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape: *
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
q
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*&
_output_shapes
: *
dtype0
�
adam/dense_9_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_9_bias_velocity/*
dtype0*
shape:2*+
shared_nameadam/dense_9_bias_velocity
�
.adam/dense_9_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_9_bias_velocity*
_output_shapes
:2*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpadam/dense_9_bias_velocity*
_class
loc:@Variable_8*
_output_shapes
:2*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:2*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
:2*
dtype0
�
adam/dense_9_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_9_bias_momentum/*
dtype0*
shape:2*+
shared_nameadam/dense_9_bias_momentum
�
.adam/dense_9_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_9_bias_momentum*
_output_shapes
:2*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpadam/dense_9_bias_momentum*
_class
loc:@Variable_9*
_output_shapes
:2*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:2*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
e
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
:2*
dtype0
�
adam/dense_9_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_9_kernel_velocity/*
dtype0*
shape:	�2*-
shared_nameadam/dense_9_kernel_velocity
�
0adam/dense_9_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_9_kernel_velocity*
_output_shapes
:	�2*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpadam/dense_9_kernel_velocity*
_class
loc:@Variable_10*
_output_shapes
:	�2*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:	�2*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
l
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
:	�2*
dtype0
�
adam/dense_9_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_9_kernel_momentum/*
dtype0*
shape:	�2*-
shared_nameadam/dense_9_kernel_momentum
�
0adam/dense_9_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_9_kernel_momentum*
_output_shapes
:	�2*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOpadam/dense_9_kernel_momentum*
_class
loc:@Variable_11*
_output_shapes
:	�2*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:	�2*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
l
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
:	�2*
dtype0
�
adam/dense_8_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_8_bias_velocity/*
dtype0*
shape:�*+
shared_nameadam/dense_8_bias_velocity
�
.adam/dense_8_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_8_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOpadam/dense_8_bias_velocity*
_class
loc:@Variable_12*
_output_shapes	
:�*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:�*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
h
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes	
:�*
dtype0
�
adam/dense_8_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_8_bias_momentum/*
dtype0*
shape:�*+
shared_nameadam/dense_8_bias_momentum
�
.adam/dense_8_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_8_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOpadam/dense_8_bias_momentum*
_class
loc:@Variable_13*
_output_shapes	
:�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
h
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes	
:�*
dtype0
�
adam/dense_8_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_8_kernel_velocity/*
dtype0*
shape:���*-
shared_nameadam/dense_8_kernel_velocity
�
0adam/dense_8_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_8_kernel_velocity*!
_output_shapes
:���*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOpadam/dense_8_kernel_velocity*
_class
loc:@Variable_14*!
_output_shapes
:���*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:���*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
n
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*!
_output_shapes
:���*
dtype0
�
adam/dense_8_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_8_kernel_momentum/*
dtype0*
shape:���*-
shared_nameadam/dense_8_kernel_momentum
�
0adam/dense_8_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_8_kernel_momentum*!
_output_shapes
:���*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOpadam/dense_8_kernel_momentum*
_class
loc:@Variable_15*!
_output_shapes
:���*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:���*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
n
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*!
_output_shapes
:���*
dtype0
�
adam/conv2d_10_bias_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/conv2d_10_bias_velocity/*
dtype0*
shape:@*-
shared_nameadam/conv2d_10_bias_velocity
�
0adam/conv2d_10_bias_velocity/Read/ReadVariableOpReadVariableOpadam/conv2d_10_bias_velocity*
_output_shapes
:@*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOpadam/conv2d_10_bias_velocity*
_class
loc:@Variable_16*
_output_shapes
:@*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:@*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
g
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes
:@*
dtype0
�
adam/conv2d_10_bias_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/conv2d_10_bias_momentum/*
dtype0*
shape:@*-
shared_nameadam/conv2d_10_bias_momentum
�
0adam/conv2d_10_bias_momentum/Read/ReadVariableOpReadVariableOpadam/conv2d_10_bias_momentum*
_output_shapes
:@*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOpadam/conv2d_10_bias_momentum*
_class
loc:@Variable_17*
_output_shapes
:@*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:@*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
g
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes
:@*
dtype0
�
adam/conv2d_10_kernel_velocityVarHandleOp*
_output_shapes
: */

debug_name!adam/conv2d_10_kernel_velocity/*
dtype0*
shape: @*/
shared_name adam/conv2d_10_kernel_velocity
�
2adam/conv2d_10_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/conv2d_10_kernel_velocity*&
_output_shapes
: @*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOpadam/conv2d_10_kernel_velocity*
_class
loc:@Variable_18*&
_output_shapes
: @*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape: @*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
s
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*&
_output_shapes
: @*
dtype0
�
adam/conv2d_10_kernel_momentumVarHandleOp*
_output_shapes
: */

debug_name!adam/conv2d_10_kernel_momentum/*
dtype0*
shape: @*/
shared_name adam/conv2d_10_kernel_momentum
�
2adam/conv2d_10_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/conv2d_10_kernel_momentum*&
_output_shapes
: @*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOpadam/conv2d_10_kernel_momentum*
_class
loc:@Variable_19*&
_output_shapes
: @*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape: @*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
s
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*&
_output_shapes
: @*
dtype0
�
adam/conv2d_9_bias_velocityVarHandleOp*
_output_shapes
: *,

debug_nameadam/conv2d_9_bias_velocity/*
dtype0*
shape: *,
shared_nameadam/conv2d_9_bias_velocity
�
/adam/conv2d_9_bias_velocity/Read/ReadVariableOpReadVariableOpadam/conv2d_9_bias_velocity*
_output_shapes
: *
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOpadam/conv2d_9_bias_velocity*
_class
loc:@Variable_20*
_output_shapes
: *
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape: *
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
g
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes
: *
dtype0
�
adam/conv2d_9_bias_momentumVarHandleOp*
_output_shapes
: *,

debug_nameadam/conv2d_9_bias_momentum/*
dtype0*
shape: *,
shared_nameadam/conv2d_9_bias_momentum
�
/adam/conv2d_9_bias_momentum/Read/ReadVariableOpReadVariableOpadam/conv2d_9_bias_momentum*
_output_shapes
: *
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOpadam/conv2d_9_bias_momentum*
_class
loc:@Variable_21*
_output_shapes
: *
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape: *
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
g
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*
_output_shapes
: *
dtype0
�
adam/conv2d_9_kernel_velocityVarHandleOp*
_output_shapes
: *.

debug_name adam/conv2d_9_kernel_velocity/*
dtype0*
shape: *.
shared_nameadam/conv2d_9_kernel_velocity
�
1adam/conv2d_9_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/conv2d_9_kernel_velocity*&
_output_shapes
: *
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOpadam/conv2d_9_kernel_velocity*
_class
loc:@Variable_22*&
_output_shapes
: *
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape: *
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
s
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*&
_output_shapes
: *
dtype0
�
adam/conv2d_9_kernel_momentumVarHandleOp*
_output_shapes
: *.

debug_name adam/conv2d_9_kernel_momentum/*
dtype0*
shape: *.
shared_nameadam/conv2d_9_kernel_momentum
�
1adam/conv2d_9_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/conv2d_9_kernel_momentum*&
_output_shapes
: *
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOpadam/conv2d_9_kernel_momentum*
_class
loc:@Variable_23*&
_output_shapes
: *
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape: *
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
s
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*&
_output_shapes
: *
dtype0
�
adam/learning_rateVarHandleOp*
_output_shapes
: *#

debug_nameadam/learning_rate/*
dtype0*
shape: *#
shared_nameadam/learning_rate
q
&adam/learning_rate/Read/ReadVariableOpReadVariableOpadam/learning_rate*
_output_shapes
: *
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOpadam/learning_rate*
_class
loc:@Variable_24*
_output_shapes
: *
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape: *
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
c
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes
: *
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
&Variable_25/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_25*
_output_shapes
: *
dtype0	
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0	*
shape: *
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0	
c
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*
_output_shapes
: *
dtype0	
�
serving_default_inputsPlaceholder*/
_output_shapes
:���������ii*
dtype0*$
shape:���������ii
�
serving_default_inputs_1Placeholder*/
_output_shapes
:���������ii*
dtype0*$
shape:���������ii
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsserving_default_inputs_1conv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *<
f7R5
3__inference_signature_wrapper_serving_default_45436

NoOpNoOp
�(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�'
value�'B�' B�'
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_operations
_layers
_build_shapes_dict
output_names
		optimizer

_default_save_signature

signatures*
* 
* 
* 
* 
 
0
1
2
3*
 
0
1
2
3*
* 
* 
�

_variables
_trainable_variables
 _trainable_variables_indices

iterations
_learning_rate

_momentums
_velocities*

trace_0* 

serving_default* 
G
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids* 
G
_inbound_nodes
_outbound_nodes
_losses
 	_loss_ids* 
�
!_tracked
"_inbound_nodes
#_outbound_nodes
$_losses
%_operations
&_layers
'_build_shapes_dict
(output_names
)_default_save_signature*
n
*_inbound_nodes
+_outbound_nodes
,_losses
-	_loss_ids
.	arguments
/_build_shapes_dict* 
�
0
1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17*
<
@0
A1
B2
C3
D4
E5
F6
G7*
* 
TN
VARIABLE_VALUEVariable_25/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEVariable_243optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
R
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10*
R
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10*
* 
* 

Strace_0* 
* 
* 
* 
* 
* 
* 
VP
VARIABLE_VALUEVariable_231optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_221optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_211optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_201optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_191optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_181optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_171optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_161optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_152optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_142optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_132optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_122optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_112optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_102optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_92optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_82optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_7;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_6;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_5;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_4;optimizer/_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_3;optimizer/_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_2;optimizer/_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_1;optimizer/_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable;optimizer/_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
G
T_inbound_nodes
U_outbound_nodes
V_losses
W	_loss_ids* 
w

@kernel
Abias
X_inbound_nodes
Y_outbound_nodes
Z_losses
[	_loss_ids
\_build_shapes_dict*
_
]_inbound_nodes
^_outbound_nodes
__losses
`	_loss_ids
a_build_shapes_dict* 
G
b_inbound_nodes
c_outbound_nodes
d_losses
e	_loss_ids* 
w

Bkernel
Cbias
f_inbound_nodes
g_outbound_nodes
h_losses
i	_loss_ids
j_build_shapes_dict*
_
k_inbound_nodes
l_outbound_nodes
m_losses
n	_loss_ids
o_build_shapes_dict* 
G
p_inbound_nodes
q_outbound_nodes
r_losses
s	_loss_ids* 
_
t_inbound_nodes
u_outbound_nodes
v_losses
w	_loss_ids
x_build_shapes_dict* 
x
D_kernel
Ebias
y_inbound_nodes
z_outbound_nodes
{_losses
|	_loss_ids
}_build_shapes_dict*
I
~_inbound_nodes
_outbound_nodes
�_losses
�	_loss_ids* 
}
F_kernel
Gbias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst*'
Tin 
2*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_45759
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*&
Tin
2*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_45846��
�v
�
!__inference__traced_restore_45846
file_prefix&
assignvariableop_variable_25:	 (
assignvariableop_1_variable_24: 8
assignvariableop_2_variable_23: 8
assignvariableop_3_variable_22: ,
assignvariableop_4_variable_21: ,
assignvariableop_5_variable_20: 8
assignvariableop_6_variable_19: @8
assignvariableop_7_variable_18: @,
assignvariableop_8_variable_17:@,
assignvariableop_9_variable_16:@4
assignvariableop_10_variable_15:���4
assignvariableop_11_variable_14:���.
assignvariableop_12_variable_13:	�.
assignvariableop_13_variable_12:	�2
assignvariableop_14_variable_11:	�22
assignvariableop_15_variable_10:	�2,
assignvariableop_16_variable_9:2,
assignvariableop_17_variable_8:28
assignvariableop_18_variable_7: ,
assignvariableop_19_variable_6: 8
assignvariableop_20_variable_5: @,
assignvariableop_21_variable_4:@3
assignvariableop_22_variable_3:���-
assignvariableop_23_variable_2:	�1
assignvariableop_24_variable_1:	�2*
assignvariableop_25_variable:2
identity_27��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_25Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_24Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_23Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_22Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_21Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_20Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_19Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_18Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_17Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_16Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_15Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_14Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_13Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_12Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_11Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_10Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_9Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_8Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_7Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_6Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_5Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_4Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_3Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_2Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variableIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_27Identity_27:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+
'
%
_user_specified_nameVariable_16:+	'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_25:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�>
�
!__inference_serving_default_45476

inputsX
>functional_17_1_conv2d_9_1_convolution_readvariableop_resource: H
:functional_17_1_conv2d_9_1_reshape_readvariableop_resource: Y
?functional_17_1_conv2d_10_1_convolution_readvariableop_resource: @I
;functional_17_1_conv2d_10_1_reshape_readvariableop_resource:@K
6functional_17_1_dense_8_1_cast_readvariableop_resource:���D
5functional_17_1_dense_8_1_add_readvariableop_resource:	�I
6functional_17_1_dense_9_1_cast_readvariableop_resource:	�2C
5functional_17_1_dense_9_1_add_readvariableop_resource:2
identity��2functional_17_1/conv2d_10_1/Reshape/ReadVariableOp�6functional_17_1/conv2d_10_1/convolution/ReadVariableOp�1functional_17_1/conv2d_9_1/Reshape/ReadVariableOp�5functional_17_1/conv2d_9_1/convolution/ReadVariableOp�,functional_17_1/dense_8_1/Add/ReadVariableOp�-functional_17_1/dense_8_1/Cast/ReadVariableOp�,functional_17_1/dense_9_1/Add/ReadVariableOp�-functional_17_1/dense_9_1/Cast/ReadVariableOp�
5functional_17_1/conv2d_9_1/convolution/ReadVariableOpReadVariableOp>functional_17_1_conv2d_9_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
&functional_17_1/conv2d_9_1/convolutionConv2Dinputs=functional_17_1/conv2d_9_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������gg *
paddingVALID*
strides
�
1functional_17_1/conv2d_9_1/Reshape/ReadVariableOpReadVariableOp:functional_17_1_conv2d_9_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0�
(functional_17_1/conv2d_9_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
"functional_17_1/conv2d_9_1/ReshapeReshape9functional_17_1/conv2d_9_1/Reshape/ReadVariableOp:value:01functional_17_1/conv2d_9_1/Reshape/shape:output:0*
T0*&
_output_shapes
: �
functional_17_1/conv2d_9_1/addAddV2/functional_17_1/conv2d_9_1/convolution:output:0+functional_17_1/conv2d_9_1/Reshape:output:0*
T0*/
_output_shapes
:���������gg �
functional_17_1/conv2d_9_1/ReluRelu"functional_17_1/conv2d_9_1/add:z:0*
T0*/
_output_shapes
:���������gg �
+functional_17_1/max_pooling2d_8_1/MaxPool2dMaxPool-functional_17_1/conv2d_9_1/Relu:activations:0*/
_output_shapes
:���������33 *
ksize
*
paddingVALID*
strides
�
6functional_17_1/conv2d_10_1/convolution/ReadVariableOpReadVariableOp?functional_17_1_conv2d_10_1_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
'functional_17_1/conv2d_10_1/convolutionConv2D4functional_17_1/max_pooling2d_8_1/MaxPool2d:output:0>functional_17_1/conv2d_10_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������11@*
paddingVALID*
strides
�
2functional_17_1/conv2d_10_1/Reshape/ReadVariableOpReadVariableOp;functional_17_1_conv2d_10_1_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0�
)functional_17_1/conv2d_10_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
#functional_17_1/conv2d_10_1/ReshapeReshape:functional_17_1/conv2d_10_1/Reshape/ReadVariableOp:value:02functional_17_1/conv2d_10_1/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
functional_17_1/conv2d_10_1/addAddV20functional_17_1/conv2d_10_1/convolution:output:0,functional_17_1/conv2d_10_1/Reshape:output:0*
T0*/
_output_shapes
:���������11@�
 functional_17_1/conv2d_10_1/ReluRelu#functional_17_1/conv2d_10_1/add:z:0*
T0*/
_output_shapes
:���������11@�
+functional_17_1/max_pooling2d_9_1/MaxPool2dMaxPool.functional_17_1/conv2d_10_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
z
)functional_17_1/flatten_4_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� �  �
#functional_17_1/flatten_4_1/ReshapeReshape4functional_17_1/max_pooling2d_9_1/MaxPool2d:output:02functional_17_1/flatten_4_1/Reshape/shape:output:0*
T0*)
_output_shapes
:������������
-functional_17_1/dense_8_1/Cast/ReadVariableOpReadVariableOp6functional_17_1_dense_8_1_cast_readvariableop_resource*!
_output_shapes
:���*
dtype0�
 functional_17_1/dense_8_1/MatMulMatMul,functional_17_1/flatten_4_1/Reshape:output:05functional_17_1/dense_8_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,functional_17_1/dense_8_1/Add/ReadVariableOpReadVariableOp5functional_17_1_dense_8_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
functional_17_1/dense_8_1/AddAddV2*functional_17_1/dense_8_1/MatMul:product:04functional_17_1/dense_8_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������|
functional_17_1/dense_8_1/ReluRelu!functional_17_1/dense_8_1/Add:z:0*
T0*(
_output_shapes
:�����������
-functional_17_1/dense_9_1/Cast/ReadVariableOpReadVariableOp6functional_17_1_dense_9_1_cast_readvariableop_resource*
_output_shapes
:	�2*
dtype0�
 functional_17_1/dense_9_1/MatMulMatMul,functional_17_1/dense_8_1/Relu:activations:05functional_17_1/dense_9_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
,functional_17_1/dense_9_1/Add/ReadVariableOpReadVariableOp5functional_17_1_dense_9_1_add_readvariableop_resource*
_output_shapes
:2*
dtype0�
functional_17_1/dense_9_1/AddAddV2*functional_17_1/dense_9_1/MatMul:product:04functional_17_1/dense_9_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2{
functional_17_1/dense_9_1/ReluRelu!functional_17_1/dense_9_1/Add:z:0*
T0*'
_output_shapes
:���������2{
IdentityIdentity,functional_17_1/dense_9_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp3^functional_17_1/conv2d_10_1/Reshape/ReadVariableOp7^functional_17_1/conv2d_10_1/convolution/ReadVariableOp2^functional_17_1/conv2d_9_1/Reshape/ReadVariableOp6^functional_17_1/conv2d_9_1/convolution/ReadVariableOp-^functional_17_1/dense_8_1/Add/ReadVariableOp.^functional_17_1/dense_8_1/Cast/ReadVariableOp-^functional_17_1/dense_9_1/Add/ReadVariableOp.^functional_17_1/dense_9_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������ii: : : : : : : : 2h
2functional_17_1/conv2d_10_1/Reshape/ReadVariableOp2functional_17_1/conv2d_10_1/Reshape/ReadVariableOp2p
6functional_17_1/conv2d_10_1/convolution/ReadVariableOp6functional_17_1/conv2d_10_1/convolution/ReadVariableOp2f
1functional_17_1/conv2d_9_1/Reshape/ReadVariableOp1functional_17_1/conv2d_9_1/Reshape/ReadVariableOp2n
5functional_17_1/conv2d_9_1/convolution/ReadVariableOp5functional_17_1/conv2d_9_1/convolution/ReadVariableOp2\
,functional_17_1/dense_8_1/Add/ReadVariableOp,functional_17_1/dense_8_1/Add/ReadVariableOp2^
-functional_17_1/dense_8_1/Cast/ReadVariableOp-functional_17_1/dense_8_1/Cast/ReadVariableOp2\
,functional_17_1/dense_9_1/Add/ReadVariableOp,functional_17_1/dense_9_1/Add/ReadVariableOp2^
-functional_17_1/dense_9_1/Cast/ReadVariableOp-functional_17_1/dense_9_1/Cast/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������ii
 
_user_specified_nameinputs
�
�
3__inference_signature_wrapper_serving_default_45436

inputs
inputs_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:���
	unknown_4:	�
	unknown_5:	�2
	unknown_6:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference_serving_default_45413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������ii:���������ii: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%	!

_user_specified_name45432:%!

_user_specified_name45430:%!

_user_specified_name45428:%!

_user_specified_name45426:%!

_user_specified_name45424:%!

_user_specified_name45422:%!

_user_specified_name45420:%!

_user_specified_name45418:YU
/
_output_shapes
:���������ii
"
_user_specified_name
inputs_1:W S
/
_output_shapes
:���������ii
 
_user_specified_nameinputs
��
�
!__inference_serving_default_45413

inputs
inputs_1h
Nfunctional_19_1_functional_17_1_conv2d_9_1_convolution_readvariableop_resource: X
Jfunctional_19_1_functional_17_1_conv2d_9_1_reshape_readvariableop_resource: i
Ofunctional_19_1_functional_17_1_conv2d_10_1_convolution_readvariableop_resource: @Y
Kfunctional_19_1_functional_17_1_conv2d_10_1_reshape_readvariableop_resource:@[
Ffunctional_19_1_functional_17_1_dense_8_1_cast_readvariableop_resource:���T
Efunctional_19_1_functional_17_1_dense_8_1_add_readvariableop_resource:	�Y
Ffunctional_19_1_functional_17_1_dense_9_1_cast_readvariableop_resource:	�2S
Efunctional_19_1_functional_17_1_dense_9_1_add_readvariableop_resource:2
identity��Bfunctional_19_1/functional_17_1/conv2d_10_1/Reshape/ReadVariableOp�Ffunctional_19_1/functional_17_1/conv2d_10_1/convolution/ReadVariableOp�Afunctional_19_1/functional_17_1/conv2d_9_1/Reshape/ReadVariableOp�Efunctional_19_1/functional_17_1/conv2d_9_1/convolution/ReadVariableOp�<functional_19_1/functional_17_1/dense_8_1/Add/ReadVariableOp�=functional_19_1/functional_17_1/dense_8_1/Cast/ReadVariableOp�<functional_19_1/functional_17_1/dense_9_1/Add/ReadVariableOp�=functional_19_1/functional_17_1/dense_9_1/Cast/ReadVariableOp�Bfunctional_19_1/functional_17_3/conv2d_10_1/Reshape/ReadVariableOp�Ffunctional_19_1/functional_17_3/conv2d_10_1/convolution/ReadVariableOp�Afunctional_19_1/functional_17_3/conv2d_9_1/Reshape/ReadVariableOp�Efunctional_19_1/functional_17_3/conv2d_9_1/convolution/ReadVariableOp�<functional_19_1/functional_17_3/dense_8_1/Add/ReadVariableOp�=functional_19_1/functional_17_3/dense_8_1/Cast/ReadVariableOp�<functional_19_1/functional_17_3/dense_9_1/Add/ReadVariableOp�=functional_19_1/functional_17_3/dense_9_1/Cast/ReadVariableOp�
Efunctional_19_1/functional_17_1/conv2d_9_1/convolution/ReadVariableOpReadVariableOpNfunctional_19_1_functional_17_1_conv2d_9_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
6functional_19_1/functional_17_1/conv2d_9_1/convolutionConv2DinputsMfunctional_19_1/functional_17_1/conv2d_9_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������gg *
paddingVALID*
strides
�
Afunctional_19_1/functional_17_1/conv2d_9_1/Reshape/ReadVariableOpReadVariableOpJfunctional_19_1_functional_17_1_conv2d_9_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0�
8functional_19_1/functional_17_1/conv2d_9_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
2functional_19_1/functional_17_1/conv2d_9_1/ReshapeReshapeIfunctional_19_1/functional_17_1/conv2d_9_1/Reshape/ReadVariableOp:value:0Afunctional_19_1/functional_17_1/conv2d_9_1/Reshape/shape:output:0*
T0*&
_output_shapes
: �
.functional_19_1/functional_17_1/conv2d_9_1/addAddV2?functional_19_1/functional_17_1/conv2d_9_1/convolution:output:0;functional_19_1/functional_17_1/conv2d_9_1/Reshape:output:0*
T0*/
_output_shapes
:���������gg �
/functional_19_1/functional_17_1/conv2d_9_1/ReluRelu2functional_19_1/functional_17_1/conv2d_9_1/add:z:0*
T0*/
_output_shapes
:���������gg �
;functional_19_1/functional_17_1/max_pooling2d_8_1/MaxPool2dMaxPool=functional_19_1/functional_17_1/conv2d_9_1/Relu:activations:0*/
_output_shapes
:���������33 *
ksize
*
paddingVALID*
strides
�
Ffunctional_19_1/functional_17_1/conv2d_10_1/convolution/ReadVariableOpReadVariableOpOfunctional_19_1_functional_17_1_conv2d_10_1_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
7functional_19_1/functional_17_1/conv2d_10_1/convolutionConv2DDfunctional_19_1/functional_17_1/max_pooling2d_8_1/MaxPool2d:output:0Nfunctional_19_1/functional_17_1/conv2d_10_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������11@*
paddingVALID*
strides
�
Bfunctional_19_1/functional_17_1/conv2d_10_1/Reshape/ReadVariableOpReadVariableOpKfunctional_19_1_functional_17_1_conv2d_10_1_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0�
9functional_19_1/functional_17_1/conv2d_10_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
3functional_19_1/functional_17_1/conv2d_10_1/ReshapeReshapeJfunctional_19_1/functional_17_1/conv2d_10_1/Reshape/ReadVariableOp:value:0Bfunctional_19_1/functional_17_1/conv2d_10_1/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
/functional_19_1/functional_17_1/conv2d_10_1/addAddV2@functional_19_1/functional_17_1/conv2d_10_1/convolution:output:0<functional_19_1/functional_17_1/conv2d_10_1/Reshape:output:0*
T0*/
_output_shapes
:���������11@�
0functional_19_1/functional_17_1/conv2d_10_1/ReluRelu3functional_19_1/functional_17_1/conv2d_10_1/add:z:0*
T0*/
_output_shapes
:���������11@�
;functional_19_1/functional_17_1/max_pooling2d_9_1/MaxPool2dMaxPool>functional_19_1/functional_17_1/conv2d_10_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
9functional_19_1/functional_17_1/flatten_4_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� �  �
3functional_19_1/functional_17_1/flatten_4_1/ReshapeReshapeDfunctional_19_1/functional_17_1/max_pooling2d_9_1/MaxPool2d:output:0Bfunctional_19_1/functional_17_1/flatten_4_1/Reshape/shape:output:0*
T0*)
_output_shapes
:������������
=functional_19_1/functional_17_1/dense_8_1/Cast/ReadVariableOpReadVariableOpFfunctional_19_1_functional_17_1_dense_8_1_cast_readvariableop_resource*!
_output_shapes
:���*
dtype0�
0functional_19_1/functional_17_1/dense_8_1/MatMulMatMul<functional_19_1/functional_17_1/flatten_4_1/Reshape:output:0Efunctional_19_1/functional_17_1/dense_8_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<functional_19_1/functional_17_1/dense_8_1/Add/ReadVariableOpReadVariableOpEfunctional_19_1_functional_17_1_dense_8_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-functional_19_1/functional_17_1/dense_8_1/AddAddV2:functional_19_1/functional_17_1/dense_8_1/MatMul:product:0Dfunctional_19_1/functional_17_1/dense_8_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.functional_19_1/functional_17_1/dense_8_1/ReluRelu1functional_19_1/functional_17_1/dense_8_1/Add:z:0*
T0*(
_output_shapes
:�����������
=functional_19_1/functional_17_1/dense_9_1/Cast/ReadVariableOpReadVariableOpFfunctional_19_1_functional_17_1_dense_9_1_cast_readvariableop_resource*
_output_shapes
:	�2*
dtype0�
0functional_19_1/functional_17_1/dense_9_1/MatMulMatMul<functional_19_1/functional_17_1/dense_8_1/Relu:activations:0Efunctional_19_1/functional_17_1/dense_9_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
<functional_19_1/functional_17_1/dense_9_1/Add/ReadVariableOpReadVariableOpEfunctional_19_1_functional_17_1_dense_9_1_add_readvariableop_resource*
_output_shapes
:2*
dtype0�
-functional_19_1/functional_17_1/dense_9_1/AddAddV2:functional_19_1/functional_17_1/dense_9_1/MatMul:product:0Dfunctional_19_1/functional_17_1/dense_9_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
.functional_19_1/functional_17_1/dense_9_1/ReluRelu1functional_19_1/functional_17_1/dense_9_1/Add:z:0*
T0*'
_output_shapes
:���������2�
Efunctional_19_1/functional_17_3/conv2d_9_1/convolution/ReadVariableOpReadVariableOpNfunctional_19_1_functional_17_1_conv2d_9_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
6functional_19_1/functional_17_3/conv2d_9_1/convolutionConv2Dinputs_1Mfunctional_19_1/functional_17_3/conv2d_9_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������gg *
paddingVALID*
strides
�
Afunctional_19_1/functional_17_3/conv2d_9_1/Reshape/ReadVariableOpReadVariableOpJfunctional_19_1_functional_17_1_conv2d_9_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0�
8functional_19_1/functional_17_3/conv2d_9_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
2functional_19_1/functional_17_3/conv2d_9_1/ReshapeReshapeIfunctional_19_1/functional_17_3/conv2d_9_1/Reshape/ReadVariableOp:value:0Afunctional_19_1/functional_17_3/conv2d_9_1/Reshape/shape:output:0*
T0*&
_output_shapes
: �
.functional_19_1/functional_17_3/conv2d_9_1/addAddV2?functional_19_1/functional_17_3/conv2d_9_1/convolution:output:0;functional_19_1/functional_17_3/conv2d_9_1/Reshape:output:0*
T0*/
_output_shapes
:���������gg �
/functional_19_1/functional_17_3/conv2d_9_1/ReluRelu2functional_19_1/functional_17_3/conv2d_9_1/add:z:0*
T0*/
_output_shapes
:���������gg �
;functional_19_1/functional_17_3/max_pooling2d_8_1/MaxPool2dMaxPool=functional_19_1/functional_17_3/conv2d_9_1/Relu:activations:0*/
_output_shapes
:���������33 *
ksize
*
paddingVALID*
strides
�
Ffunctional_19_1/functional_17_3/conv2d_10_1/convolution/ReadVariableOpReadVariableOpOfunctional_19_1_functional_17_1_conv2d_10_1_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
7functional_19_1/functional_17_3/conv2d_10_1/convolutionConv2DDfunctional_19_1/functional_17_3/max_pooling2d_8_1/MaxPool2d:output:0Nfunctional_19_1/functional_17_3/conv2d_10_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������11@*
paddingVALID*
strides
�
Bfunctional_19_1/functional_17_3/conv2d_10_1/Reshape/ReadVariableOpReadVariableOpKfunctional_19_1_functional_17_1_conv2d_10_1_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0�
9functional_19_1/functional_17_3/conv2d_10_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
3functional_19_1/functional_17_3/conv2d_10_1/ReshapeReshapeJfunctional_19_1/functional_17_3/conv2d_10_1/Reshape/ReadVariableOp:value:0Bfunctional_19_1/functional_17_3/conv2d_10_1/Reshape/shape:output:0*
T0*&
_output_shapes
:@�
/functional_19_1/functional_17_3/conv2d_10_1/addAddV2@functional_19_1/functional_17_3/conv2d_10_1/convolution:output:0<functional_19_1/functional_17_3/conv2d_10_1/Reshape:output:0*
T0*/
_output_shapes
:���������11@�
0functional_19_1/functional_17_3/conv2d_10_1/ReluRelu3functional_19_1/functional_17_3/conv2d_10_1/add:z:0*
T0*/
_output_shapes
:���������11@�
;functional_19_1/functional_17_3/max_pooling2d_9_1/MaxPool2dMaxPool>functional_19_1/functional_17_3/conv2d_10_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
9functional_19_1/functional_17_3/flatten_4_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� �  �
3functional_19_1/functional_17_3/flatten_4_1/ReshapeReshapeDfunctional_19_1/functional_17_3/max_pooling2d_9_1/MaxPool2d:output:0Bfunctional_19_1/functional_17_3/flatten_4_1/Reshape/shape:output:0*
T0*)
_output_shapes
:������������
=functional_19_1/functional_17_3/dense_8_1/Cast/ReadVariableOpReadVariableOpFfunctional_19_1_functional_17_1_dense_8_1_cast_readvariableop_resource*!
_output_shapes
:���*
dtype0�
0functional_19_1/functional_17_3/dense_8_1/MatMulMatMul<functional_19_1/functional_17_3/flatten_4_1/Reshape:output:0Efunctional_19_1/functional_17_3/dense_8_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
<functional_19_1/functional_17_3/dense_8_1/Add/ReadVariableOpReadVariableOpEfunctional_19_1_functional_17_1_dense_8_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-functional_19_1/functional_17_3/dense_8_1/AddAddV2:functional_19_1/functional_17_3/dense_8_1/MatMul:product:0Dfunctional_19_1/functional_17_3/dense_8_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.functional_19_1/functional_17_3/dense_8_1/ReluRelu1functional_19_1/functional_17_3/dense_8_1/Add:z:0*
T0*(
_output_shapes
:�����������
=functional_19_1/functional_17_3/dense_9_1/Cast/ReadVariableOpReadVariableOpFfunctional_19_1_functional_17_1_dense_9_1_cast_readvariableop_resource*
_output_shapes
:	�2*
dtype0�
0functional_19_1/functional_17_3/dense_9_1/MatMulMatMul<functional_19_1/functional_17_3/dense_8_1/Relu:activations:0Efunctional_19_1/functional_17_3/dense_9_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
<functional_19_1/functional_17_3/dense_9_1/Add/ReadVariableOpReadVariableOpEfunctional_19_1_functional_17_1_dense_9_1_add_readvariableop_resource*
_output_shapes
:2*
dtype0�
-functional_19_1/functional_17_3/dense_9_1/AddAddV2:functional_19_1/functional_17_3/dense_9_1/MatMul:product:0Dfunctional_19_1/functional_17_3/dense_9_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
.functional_19_1/functional_17_3/dense_9_1/ReluRelu1functional_19_1/functional_17_3/dense_9_1/Add:z:0*
T0*'
_output_shapes
:���������2�
functional_19_1/lambda_4_1/subSub<functional_19_1/functional_17_1/dense_9_1/Relu:activations:0<functional_19_1/functional_17_3/dense_9_1/Relu:activations:0*
T0*'
_output_shapes
:���������2�
!functional_19_1/lambda_4_1/SquareSquare"functional_19_1/lambda_4_1/sub:z:0*
T0*'
_output_shapes
:���������2r
0functional_19_1/lambda_4_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
functional_19_1/lambda_4_1/SumSum%functional_19_1/lambda_4_1/Square:y:09functional_19_1/lambda_4_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(i
$functional_19_1/lambda_4_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
"functional_19_1/lambda_4_1/MaximumMaximum'functional_19_1/lambda_4_1/Sum:output:0-functional_19_1/lambda_4_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������e
 functional_19_1/lambda_4_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
$functional_19_1/lambda_4_1/Maximum_1Maximum&functional_19_1/lambda_4_1/Maximum:z:0)functional_19_1/lambda_4_1/Const:output:0*
T0*'
_output_shapes
:����������
functional_19_1/lambda_4_1/SqrtSqrt(functional_19_1/lambda_4_1/Maximum_1:z:0*
T0*'
_output_shapes
:���������r
IdentityIdentity#functional_19_1/lambda_4_1/Sqrt:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOpC^functional_19_1/functional_17_1/conv2d_10_1/Reshape/ReadVariableOpG^functional_19_1/functional_17_1/conv2d_10_1/convolution/ReadVariableOpB^functional_19_1/functional_17_1/conv2d_9_1/Reshape/ReadVariableOpF^functional_19_1/functional_17_1/conv2d_9_1/convolution/ReadVariableOp=^functional_19_1/functional_17_1/dense_8_1/Add/ReadVariableOp>^functional_19_1/functional_17_1/dense_8_1/Cast/ReadVariableOp=^functional_19_1/functional_17_1/dense_9_1/Add/ReadVariableOp>^functional_19_1/functional_17_1/dense_9_1/Cast/ReadVariableOpC^functional_19_1/functional_17_3/conv2d_10_1/Reshape/ReadVariableOpG^functional_19_1/functional_17_3/conv2d_10_1/convolution/ReadVariableOpB^functional_19_1/functional_17_3/conv2d_9_1/Reshape/ReadVariableOpF^functional_19_1/functional_17_3/conv2d_9_1/convolution/ReadVariableOp=^functional_19_1/functional_17_3/dense_8_1/Add/ReadVariableOp>^functional_19_1/functional_17_3/dense_8_1/Cast/ReadVariableOp=^functional_19_1/functional_17_3/dense_9_1/Add/ReadVariableOp>^functional_19_1/functional_17_3/dense_9_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:���������ii:���������ii: : : : : : : : 2�
Bfunctional_19_1/functional_17_1/conv2d_10_1/Reshape/ReadVariableOpBfunctional_19_1/functional_17_1/conv2d_10_1/Reshape/ReadVariableOp2�
Ffunctional_19_1/functional_17_1/conv2d_10_1/convolution/ReadVariableOpFfunctional_19_1/functional_17_1/conv2d_10_1/convolution/ReadVariableOp2�
Afunctional_19_1/functional_17_1/conv2d_9_1/Reshape/ReadVariableOpAfunctional_19_1/functional_17_1/conv2d_9_1/Reshape/ReadVariableOp2�
Efunctional_19_1/functional_17_1/conv2d_9_1/convolution/ReadVariableOpEfunctional_19_1/functional_17_1/conv2d_9_1/convolution/ReadVariableOp2|
<functional_19_1/functional_17_1/dense_8_1/Add/ReadVariableOp<functional_19_1/functional_17_1/dense_8_1/Add/ReadVariableOp2~
=functional_19_1/functional_17_1/dense_8_1/Cast/ReadVariableOp=functional_19_1/functional_17_1/dense_8_1/Cast/ReadVariableOp2|
<functional_19_1/functional_17_1/dense_9_1/Add/ReadVariableOp<functional_19_1/functional_17_1/dense_9_1/Add/ReadVariableOp2~
=functional_19_1/functional_17_1/dense_9_1/Cast/ReadVariableOp=functional_19_1/functional_17_1/dense_9_1/Cast/ReadVariableOp2�
Bfunctional_19_1/functional_17_3/conv2d_10_1/Reshape/ReadVariableOpBfunctional_19_1/functional_17_3/conv2d_10_1/Reshape/ReadVariableOp2�
Ffunctional_19_1/functional_17_3/conv2d_10_1/convolution/ReadVariableOpFfunctional_19_1/functional_17_3/conv2d_10_1/convolution/ReadVariableOp2�
Afunctional_19_1/functional_17_3/conv2d_9_1/Reshape/ReadVariableOpAfunctional_19_1/functional_17_3/conv2d_9_1/Reshape/ReadVariableOp2�
Efunctional_19_1/functional_17_3/conv2d_9_1/convolution/ReadVariableOpEfunctional_19_1/functional_17_3/conv2d_9_1/convolution/ReadVariableOp2|
<functional_19_1/functional_17_3/dense_8_1/Add/ReadVariableOp<functional_19_1/functional_17_3/dense_8_1/Add/ReadVariableOp2~
=functional_19_1/functional_17_3/dense_8_1/Cast/ReadVariableOp=functional_19_1/functional_17_3/dense_8_1/Cast/ReadVariableOp2|
<functional_19_1/functional_17_3/dense_9_1/Add/ReadVariableOp<functional_19_1/functional_17_3/dense_9_1/Add/ReadVariableOp2~
=functional_19_1/functional_17_3/dense_9_1/Cast/ReadVariableOp=functional_19_1/functional_17_3/dense_9_1/Cast/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:WS
/
_output_shapes
:���������ii
 
_user_specified_nameinputs:W S
/
_output_shapes
:���������ii
 
_user_specified_nameinputs
��
�
__inference__traced_save_45759
file_prefix,
"read_disablecopyonread_variable_25:	 .
$read_1_disablecopyonread_variable_24: >
$read_2_disablecopyonread_variable_23: >
$read_3_disablecopyonread_variable_22: 2
$read_4_disablecopyonread_variable_21: 2
$read_5_disablecopyonread_variable_20: >
$read_6_disablecopyonread_variable_19: @>
$read_7_disablecopyonread_variable_18: @2
$read_8_disablecopyonread_variable_17:@2
$read_9_disablecopyonread_variable_16:@:
%read_10_disablecopyonread_variable_15:���:
%read_11_disablecopyonread_variable_14:���4
%read_12_disablecopyonread_variable_13:	�4
%read_13_disablecopyonread_variable_12:	�8
%read_14_disablecopyonread_variable_11:	�28
%read_15_disablecopyonread_variable_10:	�22
$read_16_disablecopyonread_variable_9:22
$read_17_disablecopyonread_variable_8:2>
$read_18_disablecopyonread_variable_7: 2
$read_19_disablecopyonread_variable_6: >
$read_20_disablecopyonread_variable_5: @2
$read_21_disablecopyonread_variable_4:@9
$read_22_disablecopyonread_variable_3:���3
$read_23_disablecopyonread_variable_2:	�7
$read_24_disablecopyonread_variable_1:	�20
"read_25_disablecopyonread_variable:2
savev2_const
identity_53��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_25*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_25^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_24*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_24^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_23*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_23^Read_2/DisableCopyOnRead*&
_output_shapes
: *
dtype0f

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*&
_output_shapes
: k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_22*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_22^Read_3/DisableCopyOnRead*&
_output_shapes
: *
dtype0f

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*&
_output_shapes
: k

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*&
_output_shapes
: i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_21*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_21^Read_4/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_20*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_20^Read_5/DisableCopyOnRead*
_output_shapes
: *
dtype0[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_19*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_19^Read_6/DisableCopyOnRead*&
_output_shapes
: @*
dtype0g
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
: @i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_18*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_18^Read_7/DisableCopyOnRead*&
_output_shapes
: @*
dtype0g
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*&
_output_shapes
: @i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_17*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_17^Read_8/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_16*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_16^Read_9/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_15*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_15^Read_10/DisableCopyOnRead*!
_output_shapes
:���*
dtype0c
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*!
_output_shapes
:���h
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*!
_output_shapes
:���k
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_14*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_14^Read_11/DisableCopyOnRead*!
_output_shapes
:���*
dtype0c
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*!
_output_shapes
:���h
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*!
_output_shapes
:���k
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_13*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_13^Read_12/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_12*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_12^Read_13/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_11*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_11^Read_14/DisableCopyOnRead*
_output_shapes
:	�2*
dtype0a
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	�2k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_10*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_10^Read_15/DisableCopyOnRead*
_output_shapes
:	�2*
dtype0a
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	�2j
Read_16/DisableCopyOnReadDisableCopyOnRead$read_16_disablecopyonread_variable_9*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp$read_16_disablecopyonread_variable_9^Read_16/DisableCopyOnRead*
_output_shapes
:2*
dtype0\
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:2a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:2j
Read_17/DisableCopyOnReadDisableCopyOnRead$read_17_disablecopyonread_variable_8*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp$read_17_disablecopyonread_variable_8^Read_17/DisableCopyOnRead*
_output_shapes
:2*
dtype0\
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes
:2a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:2j
Read_18/DisableCopyOnReadDisableCopyOnRead$read_18_disablecopyonread_variable_7*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp$read_18_disablecopyonread_variable_7^Read_18/DisableCopyOnRead*&
_output_shapes
: *
dtype0h
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*&
_output_shapes
: m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
: j
Read_19/DisableCopyOnReadDisableCopyOnRead$read_19_disablecopyonread_variable_6*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp$read_19_disablecopyonread_variable_6^Read_19/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: j
Read_20/DisableCopyOnReadDisableCopyOnRead$read_20_disablecopyonread_variable_5*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp$read_20_disablecopyonread_variable_5^Read_20/DisableCopyOnRead*&
_output_shapes
: @*
dtype0h
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
: @j
Read_21/DisableCopyOnReadDisableCopyOnRead$read_21_disablecopyonread_variable_4*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp$read_21_disablecopyonread_variable_4^Read_21/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_22/DisableCopyOnReadDisableCopyOnRead$read_22_disablecopyonread_variable_3*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp$read_22_disablecopyonread_variable_3^Read_22/DisableCopyOnRead*!
_output_shapes
:���*
dtype0c
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*!
_output_shapes
:���h
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*!
_output_shapes
:���j
Read_23/DisableCopyOnReadDisableCopyOnRead$read_23_disablecopyonread_variable_2*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp$read_23_disablecopyonread_variable_2^Read_23/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_24/DisableCopyOnReadDisableCopyOnRead$read_24_disablecopyonread_variable_1*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp$read_24_disablecopyonread_variable_1^Read_24/DisableCopyOnRead*
_output_shapes
:	�2*
dtype0a
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	�2h
Read_25/DisableCopyOnReadDisableCopyOnRead"read_25_disablecopyonread_variable*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp"read_25_disablecopyonread_variable^Read_25/DisableCopyOnRead*
_output_shapes
:2*
dtype0\
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes
:2a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:2L

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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B/optimizer/iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_52Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_53IdentityIdentity_52:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_53Identity_53:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+
'
%
_user_specified_nameVariable_16:+	'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_25:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
inputs_19
serving_default_inputs_1:0���������ii
A
inputs7
serving_default_inputs:0���������ii<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict:�C
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_operations
_layers
_build_shapes_dict
output_names
		optimizer

_default_save_signature

signatures"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�

_variables
_trainable_variables
 _trainable_variables_indices

iterations
_learning_rate

_momentums
_velocities"
_generic_user_object
�
trace_02�
!__inference_serving_default_45413�
���
FullArgSpec
args�

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
annotations� *L�I
G�D
 ����������ii
 ����������iiztrace_0
,
serving_default"
signature_map
c
_inbound_nodes
_outbound_nodes
_losses
	_loss_ids"
_generic_user_object
c
_inbound_nodes
_outbound_nodes
_losses
 	_loss_ids"
_generic_user_object
�
!_tracked
"_inbound_nodes
#_outbound_nodes
$_losses
%_operations
&_layers
'_build_shapes_dict
(output_names
)_default_save_signature"
_generic_user_object
�
*_inbound_nodes
+_outbound_nodes
,_losses
-	_loss_ids
.	arguments
/_build_shapes_dict"
_generic_user_object
�
0
1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17"
trackable_list_wrapper
X
@0
A1
B2
C3
D4
E5
F6
G7"
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 2adam/iteration
: 2adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�B�
!__inference_serving_default_45413inputsinputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
3__inference_signature_wrapper_serving_default_45436inputsinputs_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 '

kwonlyargs�
jinputs

jinputs_1
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
n
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10"
trackable_list_wrapper
n
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
Strace_02�
!__inference_serving_default_45476�
���
FullArgSpec
args�

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
annotations� *%�"
 ����������iizStrace_0
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
 "
trackable_dict_wrapper
5:3 2adam/conv2d_9_kernel_momentum
5:3 2adam/conv2d_9_kernel_velocity
':% 2adam/conv2d_9_bias_momentum
':% 2adam/conv2d_9_bias_velocity
6:4 @2adam/conv2d_10_kernel_momentum
6:4 @2adam/conv2d_10_kernel_velocity
(:&@2adam/conv2d_10_bias_momentum
(:&@2adam/conv2d_10_bias_velocity
/:-���2adam/dense_8_kernel_momentum
/:-���2adam/dense_8_kernel_velocity
':%�2adam/dense_8_bias_momentum
':%�2adam/dense_8_bias_velocity
-:+	�22adam/dense_9_kernel_momentum
-:+	�22adam/dense_9_kernel_velocity
&:$22adam/dense_9_bias_momentum
&:$22adam/dense_9_bias_velocity
):' 2conv2d_9/kernel
: 2conv2d_9/bias
*:( @2conv2d_10/kernel
:@2conv2d_10/bias
#:!���2dense_8/kernel
:�2dense_8/bias
!:	�22dense_9/kernel
:22dense_9/bias
c
T_inbound_nodes
U_outbound_nodes
V_losses
W	_loss_ids"
_generic_user_object
�

@kernel
Abias
X_inbound_nodes
Y_outbound_nodes
Z_losses
[	_loss_ids
\_build_shapes_dict"
_generic_user_object
{
]_inbound_nodes
^_outbound_nodes
__losses
`	_loss_ids
a_build_shapes_dict"
_generic_user_object
c
b_inbound_nodes
c_outbound_nodes
d_losses
e	_loss_ids"
_generic_user_object
�

Bkernel
Cbias
f_inbound_nodes
g_outbound_nodes
h_losses
i	_loss_ids
j_build_shapes_dict"
_generic_user_object
{
k_inbound_nodes
l_outbound_nodes
m_losses
n	_loss_ids
o_build_shapes_dict"
_generic_user_object
c
p_inbound_nodes
q_outbound_nodes
r_losses
s	_loss_ids"
_generic_user_object
{
t_inbound_nodes
u_outbound_nodes
v_losses
w	_loss_ids
x_build_shapes_dict"
_generic_user_object
�
D_kernel
Ebias
y_inbound_nodes
z_outbound_nodes
{_losses
|	_loss_ids
}_build_shapes_dict"
_generic_user_object
e
~_inbound_nodes
_outbound_nodes
�_losses
�	_loss_ids"
_generic_user_object
�
F_kernel
Gbias
�_inbound_nodes
�_outbound_nodes
�_losses
�	_loss_ids
�_build_shapes_dict"
_generic_user_object
�B�
!__inference_serving_default_45476inputs"�
���
FullArgSpec
args�

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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
!__inference_serving_default_45413�@ABCDEFGj�g
`�]
[�X
*�'
inputs_0���������ii
*�'
inputs_1���������ii
� "!�
unknown����������
!__inference_serving_default_45476f@ABCDEFG7�4
-�*
(�%
inputs���������ii
� "!�
unknown���������2�
3__inference_signature_wrapper_serving_default_45436�@ABCDEFGy�v
� 
o�l
6
inputs_1*�'
inputs_1���������ii
2
inputs(�%
inputs���������ii"3�0
.
output_0"�
output_0���������