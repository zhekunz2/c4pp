grammar Stan;

PRIMITIVE: 'int' | 'real' ;
COMPLEX : 'vector' | 'row_vector' | 'matrix' | 'unit_vector' | 'simplex' | 'ordered' | 'positive_ordered' | 'cholesky_factor_corr' | 'cholesky_factor_cov' | 'corr_matrix' | 'cov_matrix';

WS : [ \n\t\r]+ -> channel(HIDDEN) ;
ID : [a-zA-Z]+[a-zA-Z0-9_]* ;
NO_OP : ';' ;
OP : '+' | '-' | '/' | '*' | '^' ;
INT : [0-9]+ (('E'|'e') '-'? [0-9]+)?;
//tofix: + sign before integers?
DOUBLE :  (([0-9]? '.' [0-9]+) | ([1-9][0-9]* '.' [0-9]*)) (('E'|'e') '-'? [0-9]+)? ;
COMMENT : (((('/' '/')|'#') .*? '\n')| ('/*' .*? '*/')) -> channel(HIDDEN);

STRING : '"' ~["]* '"' ;
COMP_OP : '==' | '<' | '>' | '!=' | '&&' | '||' | '<=' | '>=' ;

arrays : '{' (INT|DOUBLE) (',' (INT|DOUBLE))* '}' ;
dtype : PRIMITIVE | COMPLEX ;
inbuilt : 'log' | 'sqrt' | 'increment_log_prob' | 'mean' | 'pow' | 'exp' | 'inv_cloglog' | 'inv_logit' | 'logit' | 'col' | 'int_step' | 'Phi' | 'log2' | 'exp2' | 'inv' | 'inv_sqrt'| 'inv_square'|  'square' |  'to_vector' | 'block' | 'diag_matrix' | 'log_sum_exp' | 'gamma_p' | 'rep_vector' | 'rep_matrix' | 'rep_array' |  'abs' |  'min' | 'max' | 'e' |  'sqrt2' |  'log10' |  'not_a_number' |  'positive_infinity' | 'negative_infinity' | 'machine_precision' | 'get_lp' | 'step' | 'is_inf' | 'is_nan' | 'fabs' | 'fdim' | 'fmin' | 'fmax' | 'fmod' | 'floor' | 'ceil' | 'round' | 'trunc' | 'cbrt' |   'hypot' | 'cos' | 'sin' | 'tan' | 'acos' | 'asin' | 'atan' | 'atan2' | 'cosh' | 'sinh' | 'tanh' | 'acosh' | 'asinh' | 'atanh' | 'erf' | 'erfc' | 'inv_Phi' | 'Phi_approx' | 'binary_log_loss' | 'owens_t' | 'inc_beta' | 'lbeta' |'tgamma' | 'lgamma' | 'digamma' | 'trigamma' | 'lmgamma' |  'gamma_q' | 'binomial_coefficient_log' | 'softmax' | 'sd' | 'log1m' | 'determinant' | 'log1p' | 'normal_log' | 'cauchy_log' | 'binomial_log' | 'lognormal_lpdf' | 'pareto_lpdf' |  'normal_cdf_log' |  'pi' | 'sum' | 'diag_pre_multiply' | 'inverse' | 'normal_cdf' | 'dot_self' | 'bernoulli_logit_lpmf' | 'if_else' | 'normal_lpdf' | 'append_row' | 'print' | 'lognormal_rng' | 'poisson_rng' | 'categorical_rng' | 'reject' | 'multiply_lower_tri_self_transpose' | 'multinomial_rng' | 'dirichlet_rng' | 'binomial_rng' | 'return' | 'rep_row_vector' | 'cov_exp_quad' | 'cholesky_decompose' | 'size' | 'rows' |'normal_rng' | 'mdivide_left_tri_low' | 'bernoulli_logit_rng' | 'multi_normal_rng' | 'log_mix' | 'poisson_log_lpmf' | 'binomial_lpmf' | 'bernoulli_lpmf' | 'binomial_logit_lpmf' | 'prod' | 'bernoulli_rng' | 'cumulative_sum' | 'categorical_log' | ID;
dim : INT | ID | expression ;
dims : '[' dim (',' dim)* ']' ;
limits : '<' ( 'lower' '=' expression ',' 'upper' '=' expression | 'lower' '=' expression | 'upper' '=' expression ) '>' ;
//decl : ( (Type ID) | (Type ID '[' ID ']') | Type limits? dims ID ) ';' ;
decl: dtype (limits)? dims? ID dims? ';'? ;
print_stmt: 'print' '(' expression  (',' expression)* ')' ';'? ;
function_call: inbuilt '(' (expression (',' expression )*)? ')' dims? ;
//transpose_exp : expression '\'' ;

function_call_stmt: function_call ';' ;
assign_stmt : expression ('<-' | '=') expression ';' ;
array_access : ID dims ;
//comp_expression : expression COMP_OP expression ;
block : '{' block* '}' | statement ;
if_stmt : 'if' '(' expression ')' block ( 'else' block )? ;
range_exp: expression ':' expression ;
for_loop_stmt : 'for' '(' ID 'in' range_exp ')' block ;
target_stmt : 'target' '+=' expression ';'  ;

expression :  expression '\''               #transpose
            | expression '^' expression     #exponop
            | expression '/' expression     #divop
            | expression './' expression    #accudivop
            | expression '*' expression     #mulop
            | expression '.*' expression    #accumulop
            | expression '+'  expression    #addop
            | expression '-' expression     #minusop
            | expression '==' expression    #eq
            | expression '<' expression     #lt
            | expression '>' expression     #gt
            | expression '!=' expression    #ne
            | expression '&&' expression    #and
            | expression '||' expression    #or
            | expression '<=' expression    #le
            | expression '>=' expression    #ge
            | expression '?' expression ':' expression   #ternary_if
            | expression '[' expression ':' expression ']' #subset
            | expression '|' expression (',' expression )*   #condition
            | arrays              #array_decl
            | '-' expression    #unary
            | '(' expression ')' #brackets
            | function_call     #function
            | array_access      #array
            | STRING            #string
            | INT               #integer
            | DOUBLE            #double
            | ID                #id_access
            | '#' expression    #comment
            ;

//distributions : 'normal' | 'cauchy' | 'student_t' | 'double_exponential' | 'logistic' | 'gumbel' | 'lognormal' | 'chi_square' | 'inv_chi_square' | 'exponential' | 'gamma' | 'inv_gamma' | 'weibull' | 'beta' | 'uniform' | 'bernoulli_logit' | 'binomial' | 'beta_binomial' | 'neg_binomial' | 'poisson' ;
distribution_exp : ID '(' expression (',' expression )* ')' ;
sample : expression '~' distribution_exp ';' ;
return_or_param_type: dtype | 'void' | ( dtype ('[' ']')+ ) ;
params : return_or_param_type ID  (',' return_or_param_type ID)* ;
function_decl: return_or_param_type ID '(' params? ')' block;
return_stmt : 'return' expression ';' ;
statement : NO_OP | sample | decl | print_stmt | function_call_stmt | assign_stmt | if_stmt | for_loop_stmt | return_stmt | target_stmt;


datablk : 'data'  '{' decl* '}' ;
paramblk : 'parameters' '{' decl* '}' ;
modelblk : 'model' block ;
transformed_param_blk : 'transformed parameters' block;
transformed_data_blk : 'transformed data' block;
generated_quantities_blk : 'generated quantities' block;
functions_blk : 'functions' '{' function_decl* '}';

program : (datablk | paramblk | modelblk | transformed_param_blk | transformed_data_blk | generated_quantities_blk | functions_blk)+;

