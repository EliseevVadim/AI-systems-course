��
��
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
0
Sigmoid
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-15-g6290819256d8�
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
�
Adam/v/dense_109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/dense_109/bias
{
)Adam/v/dense_109/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_109/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/dense_109/bias
{
)Adam/m/dense_109/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_109/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameAdam/v/dense_109/kernel
�
+Adam/v/dense_109/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_109/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameAdam/m/dense_109/kernel
�
+Adam/m/dense_109/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_109/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_108/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/v/dense_108/bias
|
)Adam/v/dense_108/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_108/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_108/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/m/dense_108/bias
|
)Adam/m/dense_108/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_108/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_108/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameAdam/v/dense_108/kernel
�
+Adam/v/dense_108/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_108/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_108/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_nameAdam/m/dense_108/kernel
�
+Adam/m/dense_108/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_108/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
t
dense_109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_109/bias
m
"dense_109/bias/Read/ReadVariableOpReadVariableOpdense_109/bias*
_output_shapes
:*
dtype0
}
dense_109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*!
shared_namedense_109/kernel
v
$dense_109/kernel/Read/ReadVariableOpReadVariableOpdense_109/kernel*
_output_shapes
:	�*
dtype0
u
dense_108/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_108/bias
n
"dense_108/bias/Read/ReadVariableOpReadVariableOpdense_108/bias*
_output_shapes	
:�*
dtype0
}
dense_108/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*!
shared_namedense_108/kernel
v
$dense_108/kernel/Read/ReadVariableOpReadVariableOpdense_108/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_dense_108_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_108_inputdense_108/kerneldense_108/biasdense_109/kerneldense_109/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_420743

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
!trace_0
"trace_1
#trace_2
$trace_3* 
6
%trace_0
&trace_1
'trace_2
(trace_3* 
* 
�
)
_variables
*_iterations
+_learning_rate
,_index_dict
-
_momentums
._velocities
/_update_step_xla*

0serving_default* 

0
1*

0
1*
* 
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

6trace_0* 

7trace_0* 
`Z
VARIABLE_VALUEdense_108/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_108/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

=trace_0* 

>trace_0* 
`Z
VARIABLE_VALUEdense_109/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_109/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

?0*
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
C
*0
@1
A2
B3
C4
D5
E6
F7
G8*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
@0
B1
D2
F3*
 
A0
C1
E2
G3*
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
H	variables
I	keras_api
	Jtotal
	Kcount*
b\
VARIABLE_VALUEAdam/m/dense_108/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_108/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_108/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_108/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_109/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_109/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_109/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_109/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*

J0
K1*

H	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_108/kernel/Read/ReadVariableOp"dense_108/bias/Read/ReadVariableOp$dense_109/kernel/Read/ReadVariableOp"dense_109/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp+Adam/m/dense_108/kernel/Read/ReadVariableOp+Adam/v/dense_108/kernel/Read/ReadVariableOp)Adam/m/dense_108/bias/Read/ReadVariableOp)Adam/v/dense_108/bias/Read/ReadVariableOp+Adam/m/dense_109/kernel/Read/ReadVariableOp+Adam/v/dense_109/kernel/Read/ReadVariableOp)Adam/m/dense_109/bias/Read/ReadVariableOp)Adam/v/dense_109/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
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
__inference__traced_save_420913
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_108/kerneldense_108/biasdense_109/kerneldense_109/bias	iterationlearning_rateAdam/m/dense_108/kernelAdam/v/dense_108/kernelAdam/m/dense_108/biasAdam/v/dense_108/biasAdam/m/dense_109/kernelAdam/v/dense_109/kernelAdam/m/dense_109/biasAdam/v/dense_109/biastotalcount*
Tin
2*
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
"__inference__traced_restore_420971��
�
�
.__inference_sequential_54_layer_call_fn_420756

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_54_layer_call_and_return_conditional_losses_420614o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_109_layer_call_and_return_conditional_losses_420842

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_54_layer_call_fn_420698
dense_108_input
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_108_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_54_layer_call_and_return_conditional_losses_420674o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_108_input
�
�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420712
dense_108_input#
dense_108_420701:	�
dense_108_420703:	�#
dense_109_420706:	�
dense_109_420708:
identity��!dense_108/StatefulPartitionedCall�!dense_109/StatefulPartitionedCall�
!dense_108/StatefulPartitionedCallStatefulPartitionedCalldense_108_inputdense_108_420701dense_108_420703*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_420591�
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_420706dense_109_420708*
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
GPU 2J 8� *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_420607y
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_108_input
�)
�
__inference__traced_save_420913
file_prefix/
+savev2_dense_108_kernel_read_readvariableop-
)savev2_dense_108_bias_read_readvariableop/
+savev2_dense_109_kernel_read_readvariableop-
)savev2_dense_109_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop6
2savev2_adam_m_dense_108_kernel_read_readvariableop6
2savev2_adam_v_dense_108_kernel_read_readvariableop4
0savev2_adam_m_dense_108_bias_read_readvariableop4
0savev2_adam_v_dense_108_bias_read_readvariableop6
2savev2_adam_m_dense_109_kernel_read_readvariableop6
2savev2_adam_v_dense_109_kernel_read_readvariableop4
0savev2_adam_m_dense_109_bias_read_readvariableop4
0savev2_adam_v_dense_109_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_108_kernel_read_readvariableop)savev2_dense_108_bias_read_readvariableop+savev2_dense_109_kernel_read_readvariableop)savev2_dense_109_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop2savev2_adam_m_dense_108_kernel_read_readvariableop2savev2_adam_v_dense_108_kernel_read_readvariableop0savev2_adam_m_dense_108_bias_read_readvariableop0savev2_adam_v_dense_108_bias_read_readvariableop2savev2_adam_m_dense_109_kernel_read_readvariableop2savev2_adam_v_dense_109_kernel_read_readvariableop0savev2_adam_m_dense_109_bias_read_readvariableop0savev2_adam_v_dense_109_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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
_input_shapesw
u: :	�:�:	�:: : :	�:	�:�:�:	�:	�::: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%!

_output_shapes
:	�:!	

_output_shapes	
:�:!


_output_shapes	
:�:%!

_output_shapes
:	�:%!

_output_shapes
:	�: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
.__inference_sequential_54_layer_call_fn_420769

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_54_layer_call_and_return_conditional_losses_420674o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420674

inputs#
dense_108_420663:	�
dense_108_420665:	�#
dense_109_420668:	�
dense_109_420670:
identity��!dense_108/StatefulPartitionedCall�!dense_109/StatefulPartitionedCall�
!dense_108/StatefulPartitionedCallStatefulPartitionedCallinputsdense_108_420663dense_108_420665*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_420591�
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_420668dense_109_420670*
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
GPU 2J 8� *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_420607y
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_108_layer_call_and_return_conditional_losses_420591

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_109_layer_call_and_return_conditional_losses_420607

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_109_layer_call_fn_420832

inputs
unknown:	�
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
GPU 2J 8� *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_420607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420786

inputs;
(dense_108_matmul_readvariableop_resource:	�8
)dense_108_biasadd_readvariableop_resource:	�;
(dense_109_matmul_readvariableop_resource:	�7
)dense_109_biasadd_readvariableop_resource:
identity�� dense_108/BiasAdd/ReadVariableOp�dense_108/MatMul/ReadVariableOp� dense_109/BiasAdd/ReadVariableOp�dense_109/MatMul/ReadVariableOp�
dense_108/MatMul/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0~
dense_108/MatMulMatMulinputs'dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_108/BiasAdd/ReadVariableOpReadVariableOp)dense_108_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_108/BiasAddBiasAdddense_108/MatMul:product:0(dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
dense_108/SigmoidSigmoiddense_108/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_109/MatMulMatMuldense_108/Sigmoid:y:0'dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_109/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2D
 dense_108/BiasAdd/ReadVariableOp dense_108/BiasAdd/ReadVariableOp2B
dense_108/MatMul/ReadVariableOpdense_108/MatMul/ReadVariableOp2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420803

inputs;
(dense_108_matmul_readvariableop_resource:	�8
)dense_108_biasadd_readvariableop_resource:	�;
(dense_109_matmul_readvariableop_resource:	�7
)dense_109_biasadd_readvariableop_resource:
identity�� dense_108/BiasAdd/ReadVariableOp�dense_108/MatMul/ReadVariableOp� dense_109/BiasAdd/ReadVariableOp�dense_109/MatMul/ReadVariableOp�
dense_108/MatMul/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0~
dense_108/MatMulMatMulinputs'dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 dense_108/BiasAdd/ReadVariableOpReadVariableOp)dense_108_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_108/BiasAddBiasAdddense_108/MatMul:product:0(dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������k
dense_108/SigmoidSigmoiddense_108/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_109/MatMulMatMuldense_108/Sigmoid:y:0'dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_109/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2D
 dense_108/BiasAdd/ReadVariableOp dense_108/BiasAdd/ReadVariableOp2B
dense_108/MatMul/ReadVariableOpdense_108/MatMul/ReadVariableOp2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
.__inference_sequential_54_layer_call_fn_420625
dense_108_input
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_108_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_54_layer_call_and_return_conditional_losses_420614o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_108_input
�
�
!__inference__wrapped_model_420573
dense_108_inputI
6sequential_54_dense_108_matmul_readvariableop_resource:	�F
7sequential_54_dense_108_biasadd_readvariableop_resource:	�I
6sequential_54_dense_109_matmul_readvariableop_resource:	�E
7sequential_54_dense_109_biasadd_readvariableop_resource:
identity��.sequential_54/dense_108/BiasAdd/ReadVariableOp�-sequential_54/dense_108/MatMul/ReadVariableOp�.sequential_54/dense_109/BiasAdd/ReadVariableOp�-sequential_54/dense_109/MatMul/ReadVariableOp�
-sequential_54/dense_108/MatMul/ReadVariableOpReadVariableOp6sequential_54_dense_108_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_54/dense_108/MatMulMatMuldense_108_input5sequential_54/dense_108/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_54/dense_108/BiasAdd/ReadVariableOpReadVariableOp7sequential_54_dense_108_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_54/dense_108/BiasAddBiasAdd(sequential_54/dense_108/MatMul:product:06sequential_54/dense_108/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_54/dense_108/SigmoidSigmoid(sequential_54/dense_108/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_54/dense_109/MatMul/ReadVariableOpReadVariableOp6sequential_54_dense_109_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_54/dense_109/MatMulMatMul#sequential_54/dense_108/Sigmoid:y:05sequential_54/dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_54/dense_109/BiasAdd/ReadVariableOpReadVariableOp7sequential_54_dense_109_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_54/dense_109/BiasAddBiasAdd(sequential_54/dense_109/MatMul:product:06sequential_54/dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(sequential_54/dense_109/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^sequential_54/dense_108/BiasAdd/ReadVariableOp.^sequential_54/dense_108/MatMul/ReadVariableOp/^sequential_54/dense_109/BiasAdd/ReadVariableOp.^sequential_54/dense_109/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2`
.sequential_54/dense_108/BiasAdd/ReadVariableOp.sequential_54/dense_108/BiasAdd/ReadVariableOp2^
-sequential_54/dense_108/MatMul/ReadVariableOp-sequential_54/dense_108/MatMul/ReadVariableOp2`
.sequential_54/dense_109/BiasAdd/ReadVariableOp.sequential_54/dense_109/BiasAdd/ReadVariableOp2^
-sequential_54/dense_109/MatMul/ReadVariableOp-sequential_54/dense_109/MatMul/ReadVariableOp:X T
'
_output_shapes
:���������
)
_user_specified_namedense_108_input
�
�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420614

inputs#
dense_108_420592:	�
dense_108_420594:	�#
dense_109_420608:	�
dense_109_420610:
identity��!dense_108/StatefulPartitionedCall�!dense_109/StatefulPartitionedCall�
!dense_108/StatefulPartitionedCallStatefulPartitionedCallinputsdense_108_420592dense_108_420594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_420591�
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_420608dense_109_420610*
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
GPU 2J 8� *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_420607y
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420726
dense_108_input#
dense_108_420715:	�
dense_108_420717:	�#
dense_109_420720:	�
dense_109_420722:
identity��!dense_108/StatefulPartitionedCall�!dense_109/StatefulPartitionedCall�
!dense_108/StatefulPartitionedCallStatefulPartitionedCalldense_108_inputdense_108_420715dense_108_420717*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_420591�
!dense_109/StatefulPartitionedCallStatefulPartitionedCall*dense_108/StatefulPartitionedCall:output:0dense_109_420720dense_109_420722*
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
GPU 2J 8� *N
fIRG
E__inference_dense_109_layer_call_and_return_conditional_losses_420607y
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_108/StatefulPartitionedCall"^dense_109/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_108_input
�
�
$__inference_signature_wrapper_420743
dense_108_input
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_108_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_420573o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_108_input
�
�
*__inference_dense_108_layer_call_fn_420812

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_108_layer_call_and_return_conditional_losses_420591p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_108_layer_call_and_return_conditional_losses_420823

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�G
�	
"__inference__traced_restore_420971
file_prefix4
!assignvariableop_dense_108_kernel:	�0
!assignvariableop_1_dense_108_bias:	�6
#assignvariableop_2_dense_109_kernel:	�/
!assignvariableop_3_dense_109_bias:&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: =
*assignvariableop_6_adam_m_dense_108_kernel:	�=
*assignvariableop_7_adam_v_dense_108_kernel:	�7
(assignvariableop_8_adam_m_dense_108_bias:	�7
(assignvariableop_9_adam_v_dense_108_bias:	�>
+assignvariableop_10_adam_m_dense_109_kernel:	�>
+assignvariableop_11_adam_v_dense_109_kernel:	�7
)assignvariableop_12_adam_m_dense_109_bias:7
)assignvariableop_13_adam_v_dense_109_bias:#
assignvariableop_14_total: #
assignvariableop_15_count: 
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_dense_108_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_108_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_109_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_109_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp*assignvariableop_6_adam_m_dense_108_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp*assignvariableop_7_adam_v_dense_108_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp(assignvariableop_8_adam_m_dense_108_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_v_dense_108_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_adam_m_dense_109_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp+assignvariableop_11_adam_v_dense_109_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_m_dense_109_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_v_dense_109_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
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
_user_specified_namefile_prefix"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
dense_108_input8
!serving_default_dense_108_input:0���������=
	dense_1090
StatefulPartitionedCall:0���������tensorflow/serving/predict:�]
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
!trace_0
"trace_1
#trace_2
$trace_32�
.__inference_sequential_54_layer_call_fn_420625
.__inference_sequential_54_layer_call_fn_420756
.__inference_sequential_54_layer_call_fn_420769
.__inference_sequential_54_layer_call_fn_420698�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z!trace_0z"trace_1z#trace_2z$trace_3
�
%trace_0
&trace_1
'trace_2
(trace_32�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420786
I__inference_sequential_54_layer_call_and_return_conditional_losses_420803
I__inference_sequential_54_layer_call_and_return_conditional_losses_420712
I__inference_sequential_54_layer_call_and_return_conditional_losses_420726�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z%trace_0z&trace_1z'trace_2z(trace_3
�B�
!__inference__wrapped_model_420573dense_108_input"�
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
)
_variables
*_iterations
+_learning_rate
,_index_dict
-
_momentums
._velocities
/_update_step_xla"
experimentalOptimizer
,
0serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
6trace_02�
*__inference_dense_108_layer_call_fn_420812�
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
 z6trace_0
�
7trace_02�
E__inference_dense_108_layer_call_and_return_conditional_losses_420823�
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
 z7trace_0
#:!	�2dense_108/kernel
:�2dense_108/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
=trace_02�
*__inference_dense_109_layer_call_fn_420832�
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
 z=trace_0
�
>trace_02�
E__inference_dense_109_layer_call_and_return_conditional_losses_420842�
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
 z>trace_0
#:!	�2dense_109/kernel
:2dense_109/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_54_layer_call_fn_420625dense_108_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_54_layer_call_fn_420756inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_54_layer_call_fn_420769inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_54_layer_call_fn_420698dense_108_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420786inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420803inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420712dense_108_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_54_layer_call_and_return_conditional_losses_420726dense_108_input"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
_
*0
@1
A2
B3
C4
D5
E6
F7
G8"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
<
@0
B1
D2
F3"
trackable_list_wrapper
<
A0
C1
E2
G3"
trackable_list_wrapper
�2��
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
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
 0
�B�
$__inference_signature_wrapper_420743dense_108_input"�
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
*__inference_dense_108_layer_call_fn_420812inputs"�
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
E__inference_dense_108_layer_call_and_return_conditional_losses_420823inputs"�
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
*__inference_dense_109_layer_call_fn_420832inputs"�
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
E__inference_dense_109_layer_call_and_return_conditional_losses_420842inputs"�
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
H	variables
I	keras_api
	Jtotal
	Kcount"
_tf_keras_metric
(:&	�2Adam/m/dense_108/kernel
(:&	�2Adam/v/dense_108/kernel
": �2Adam/m/dense_108/bias
": �2Adam/v/dense_108/bias
(:&	�2Adam/m/dense_109/kernel
(:&	�2Adam/v/dense_109/kernel
!:2Adam/m/dense_109/bias
!:2Adam/v/dense_109/bias
.
J0
K1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
:  (2total
:  (2count�
!__inference__wrapped_model_420573w8�5
.�+
)�&
dense_108_input���������
� "5�2
0
	dense_109#� 
	dense_109����������
E__inference_dense_108_layer_call_and_return_conditional_losses_420823d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_108_layer_call_fn_420812Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dense_109_layer_call_and_return_conditional_losses_420842d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
*__inference_dense_109_layer_call_fn_420832Y0�-
&�#
!�
inputs����������
� "!�
unknown����������
I__inference_sequential_54_layer_call_and_return_conditional_losses_420712v@�=
6�3
)�&
dense_108_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_54_layer_call_and_return_conditional_losses_420726v@�=
6�3
)�&
dense_108_input���������
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_54_layer_call_and_return_conditional_losses_420786m7�4
-�*
 �
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_54_layer_call_and_return_conditional_losses_420803m7�4
-�*
 �
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
.__inference_sequential_54_layer_call_fn_420625k@�=
6�3
)�&
dense_108_input���������
p 

 
� "!�
unknown����������
.__inference_sequential_54_layer_call_fn_420698k@�=
6�3
)�&
dense_108_input���������
p

 
� "!�
unknown����������
.__inference_sequential_54_layer_call_fn_420756b7�4
-�*
 �
inputs���������
p 

 
� "!�
unknown����������
.__inference_sequential_54_layer_call_fn_420769b7�4
-�*
 �
inputs���������
p

 
� "!�
unknown����������
$__inference_signature_wrapper_420743�K�H
� 
A�>
<
dense_108_input)�&
dense_108_input���������"5�2
0
	dense_109#� 
	dense_109���������