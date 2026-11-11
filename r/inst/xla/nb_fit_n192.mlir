module @jit_nb_fit attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x192xf64>, %arg1: tensor<192x12xf64>, %arg2: tensor<192xf64>, %arg3: tensor<192xf64>, %arg4: tensor<256xf64>) -> (tensor<256x12xf64> {jax.result_info = "result[0]"}, tensor<256xf64> {jax.result_info = "result[1]"}, tensor<256xi1> {jax.result_info = "result[2]"}, tensor<256xi32> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf64>) -> tensor<256x192xf64>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<192xf64>) -> tensor<256x192xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<256x12xf64>
    %3:4 = call @_fit_batched(%arg0, %arg1, %0, %1, %arg4, %2) : (tensor<256x192xf64>, tensor<192x12xf64>, tensor<256x192xf64>, tensor<256x192xf64>, tensor<256xf64>, tensor<256x12xf64>) -> (tensor<256x12xf64>, tensor<256xf64>, tensor<256xi64>, tensor<256xi1>)
    %4 = stablehlo.convert %3#2 : (tensor<256xi64>) -> tensor<256xi32>
    return %3#0, %3#1, %3#3, %4 : tensor<256x12xf64>, tensor<256xf64>, tensor<256xi1>, tensor<256xi32>
  }
  func.func private @_fit_batched(%arg0: tensor<256x192xf64>, %arg1: tensor<192x12xf64>, %arg2: tensor<256x192xf64>, %arg3: tensor<256x192xf64>, %arg4: tensor<256xf64>, %arg5: tensor<256x12xf64>) -> (tensor<256x12xf64>, tensor<256xf64>, tensor<256xi64>, tensor<256xi1>) {
    %c = stablehlo.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 11]> : tensor<78xi64>
    %c_0 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_1 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 7, 8, 9, 10, 11, 8, 9, 10, 11, 9, 10, 11, 10, 11, 11]> : tensor<78xi64>
    %c_2 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_3 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_4 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_5 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_6 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_7 = stablehlo.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 11]> : tensor<78xi64>
    %c_8 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_9 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 7, 8, 9, 10, 11, 8, 9, 10, 11, 9, 10, 11, 10, 11, 11]> : tensor<78xi64>
    %c_10 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_11 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_12 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_13 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_14 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_15 = stablehlo.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 11]> : tensor<78xi64>
    %c_16 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_17 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 7, 8, 9, 10, 11, 8, 9, 10, 11, 9, 10, 11, 10, 11, 11]> : tensor<78xi64>
    %c_18 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_19 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_20 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_21 = stablehlo.constant dense<false> : tensor<78xi1>
    %c_22 = stablehlo.constant dense<false> : tensor<78xi1>
    %0 = stablehlo.reshape %arg0 : (tensor<256x192xf64>) -> tensor<1x256x192xf64>
    %1 = stablehlo.reshape %arg2 : (tensor<256x192xf64>) -> tensor<1x256x192xf64>
    %2 = stablehlo.reshape %arg3 : (tensor<256x192xf64>) -> tensor<1x256x192xf64>
    %3 = stablehlo.reshape %arg4 : (tensor<256xf64>) -> tensor<1x256xf64>
    %4 = stablehlo.reshape %arg5 : (tensor<256x12xf64>) -> tensor<1x256x12xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1x256x12xf64>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %6 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f64>) -> tensor<1x256xf64>
    %c_24 = stablehlo.constant dense<0> : tensor<i64>
    %7 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i64>) -> tensor<1x256xi64>
    %c_25 = stablehlo.constant dense<false> : tensor<i1>
    %8 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i1>) -> tensor<1x256xi1>
    %c_26 = stablehlo.constant dense<0> : tensor<i64>
    %9:27 = stablehlo.while(%iterArg = %0, %iterArg_27 = %1, %iterArg_28 = %2, %iterArg_29 = %3, %iterArg_30 = %4, %iterArg_31 = %arg1, %iterArg_32 = %c, %iterArg_33 = %c_0, %iterArg_34 = %c_1, %iterArg_35 = %c_2, %iterArg_36 = %c_3, %iterArg_37 = %c_4, %iterArg_38 = %c_5, %iterArg_39 = %c_6, %iterArg_40 = %c_7, %iterArg_41 = %c_8, %iterArg_42 = %c_9, %iterArg_43 = %c_10, %iterArg_44 = %c_11, %iterArg_45 = %c_12, %iterArg_46 = %c_13, %iterArg_47 = %c_14, %iterArg_48 = %c_26, %iterArg_49 = %5, %iterArg_50 = %6, %iterArg_51 = %7, %iterArg_52 = %8) : tensor<1x256x192xf64>, tensor<1x256x192xf64>, tensor<1x256x192xf64>, tensor<1x256xf64>, tensor<1x256x12xf64>, tensor<192x12xf64>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<i64>, tensor<1x256x12xf64>, tensor<1x256xf64>, tensor<1x256xi64>, tensor<1x256xi1>
    cond {
      %c_53 = stablehlo.constant dense<1> : tensor<i64>
      %14 = stablehlo.compare LT, %iterArg_48, %c_53, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14 = func.call @dynamic_index_in_dim(%iterArg, %iterArg_48) : (tensor<1x256x192xf64>, tensor<i64>) -> tensor<256x192xf64>
      %15 = func.call @dynamic_index_in_dim(%iterArg_27, %iterArg_48) : (tensor<1x256x192xf64>, tensor<i64>) -> tensor<256x192xf64>
      %16 = func.call @dynamic_index_in_dim(%iterArg_28, %iterArg_48) : (tensor<1x256x192xf64>, tensor<i64>) -> tensor<256x192xf64>
      %17 = func.call @dynamic_index_in_dim_0(%iterArg_29, %iterArg_48) : (tensor<1x256xf64>, tensor<i64>) -> tensor<256xf64>
      %18 = func.call @dynamic_index_in_dim_1(%iterArg_30, %iterArg_48) : (tensor<1x256x12xf64>, tensor<i64>) -> tensor<256x12xf64>
      %cst_53 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
      %19 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
      %20 = stablehlo.add %14, %19 : tensor<256x192xf64>
      %21 = stablehlo.log %20 : tensor<256x192xf64>
      %22 = stablehlo.dot_general %iterArg_31, %18, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<192x12xf64>, tensor<256x12xf64>) -> tensor<192x256xf64>
      %23 = stablehlo.transpose %22, dims = [1, 0] : (tensor<192x256xf64>) -> tensor<256x192xf64>
      %24 = stablehlo.add %23, %15 : tensor<256x192xf64>
      %c_54 = stablehlo.constant dense<false> : tensor<i1>
      %25 = func.call @_where(%c_54, %24, %21) : (tensor<i1>, tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xf64>
      %26 = stablehlo.exponential %25 : tensor<256x192xf64>
      %cst_55 = stablehlo.constant dense<1.000000e-300> : tensor<f64>
      %27 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
      %28 = stablehlo.maximum %26, %27 : tensor<256x192xf64>
      %29 = stablehlo.exponential %25 : tensor<256x192xf64>
      %30 = stablehlo.multiply %28, %28 : tensor<256x192xf64>
      %31 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
      %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
      %33 = stablehlo.multiply %32, %30 : tensor<256x192xf64>
      %34 = stablehlo.add %28, %33 : tensor<256x192xf64>
      %35 = stablehlo.multiply %29, %29 : tensor<256x192xf64>
      %36 = stablehlo.multiply %16, %35 : tensor<256x192xf64>
      %37 = stablehlo.divide %36, %34 : tensor<256x192xf64>
      %38 = stablehlo.subtract %14, %28 : tensor<256x192xf64>
      %39 = stablehlo.divide %38, %29 : tensor<256x192xf64>
      %40 = stablehlo.add %25, %39 : tensor<256x192xf64>
      %41 = stablehlo.subtract %40, %15 : tensor<256x192xf64>
      %c_56 = stablehlo.constant dense<12> : tensor<i64>
      %42 = stablehlo.broadcast_in_dim %c_56, dims = [] : (tensor<i64>) -> tensor<78xi64>
      %43 = stablehlo.add %iterArg_32, %42 : tensor<78xi64>
      %44 = stablehlo.select %iterArg_33, %43, %iterArg_32 : tensor<78xi1>, tensor<78xi64>
      %45 = stablehlo.convert %44 : (tensor<78xi64>) -> tensor<78xi32>
      %46 = stablehlo.broadcast_in_dim %45, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
      %47 = "stablehlo.gather"(%iterArg_31, %46) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 192, 1>}> : (tensor<192x12xf64>, tensor<78x1xi32>) -> tensor<192x78xf64>
      %c_57 = stablehlo.constant dense<12> : tensor<i64>
      %48 = stablehlo.broadcast_in_dim %c_57, dims = [] : (tensor<i64>) -> tensor<78xi64>
      %49 = stablehlo.add %iterArg_34, %48 : tensor<78xi64>
      %50 = stablehlo.select %iterArg_35, %49, %iterArg_34 : tensor<78xi1>, tensor<78xi64>
      %51 = stablehlo.convert %50 : (tensor<78xi64>) -> tensor<78xi32>
      %52 = stablehlo.broadcast_in_dim %51, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
      %53 = "stablehlo.gather"(%iterArg_31, %52) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 192, 1>}> : (tensor<192x12xf64>, tensor<78x1xi32>) -> tensor<192x78xf64>
      %54 = stablehlo.multiply %47, %53 : tensor<192x78xf64>
      %55 = stablehlo.dot_general %37, %54, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x192xf64>, tensor<192x78xf64>) -> tensor<256x78xf64>
      %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %56 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f64>) -> tensor<12x12xf64>
      %c_59 = stablehlo.constant dense<12> : tensor<i64>
      %57 = stablehlo.broadcast_in_dim %c_59, dims = [] : (tensor<i64>) -> tensor<78xi64>
      %58 = stablehlo.add %iterArg_32, %57 : tensor<78xi64>
      %59 = stablehlo.select %iterArg_36, %58, %iterArg_32 : tensor<78xi1>, tensor<78xi64>
      %c_60 = stablehlo.constant dense<12> : tensor<i64>
      %60 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i64>) -> tensor<78xi64>
      %61 = stablehlo.add %iterArg_34, %60 : tensor<78xi64>
      %62 = stablehlo.select %iterArg_37, %61, %iterArg_34 : tensor<78xi1>, tensor<78xi64>
      %63 = stablehlo.convert %59 : (tensor<78xi64>) -> tensor<78xi32>
      %64 = stablehlo.convert %62 : (tensor<78xi64>) -> tensor<78xi32>
      %65 = stablehlo.broadcast_in_dim %63, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
      %66 = stablehlo.broadcast_in_dim %64, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
      %67 = stablehlo.concatenate %65, %66, dim = 1 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x2xi32>
      %68 = stablehlo.broadcast_in_dim %56, dims = [1, 2] : (tensor<12x12xf64>) -> tensor<256x12x12xf64>
      %69 = "stablehlo.scatter"(%68, %67, %55) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
        stablehlo.return %arg7 : tensor<f64>
      }) : (tensor<256x12x12xf64>, tensor<78x2xi32>, tensor<256x78xf64>) -> tensor<256x12x12xf64>
      %c_61 = stablehlo.constant dense<12> : tensor<i64>
      %70 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i64>) -> tensor<78xi64>
      %71 = stablehlo.add %iterArg_34, %70 : tensor<78xi64>
      %72 = stablehlo.select %iterArg_38, %71, %iterArg_34 : tensor<78xi1>, tensor<78xi64>
      %c_62 = stablehlo.constant dense<12> : tensor<i64>
      %73 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i64>) -> tensor<78xi64>
      %74 = stablehlo.add %iterArg_32, %73 : tensor<78xi64>
      %75 = stablehlo.select %iterArg_39, %74, %iterArg_32 : tensor<78xi1>, tensor<78xi64>
      %76 = stablehlo.convert %72 : (tensor<78xi64>) -> tensor<78xi32>
      %77 = stablehlo.convert %75 : (tensor<78xi64>) -> tensor<78xi32>
      %78 = stablehlo.broadcast_in_dim %76, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
      %79 = stablehlo.broadcast_in_dim %77, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
      %80 = stablehlo.concatenate %78, %79, dim = 1 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x2xi32>
      %81 = "stablehlo.scatter"(%69, %80, %55) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
        stablehlo.return %arg7 : tensor<f64>
      }) : (tensor<256x12x12xf64>, tensor<78x2xi32>, tensor<256x78xf64>) -> tensor<256x12x12xf64>
      %82 = stablehlo.iota dim = 0 : tensor<12x12xi64>
      %83 = stablehlo.iota dim = 1 : tensor<12x12xi64>
      %c_63 = stablehlo.constant dense<0> : tensor<i64>
      %84 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i64>) -> tensor<12x12xi64>
      %85 = stablehlo.add %82, %84 : tensor<12x12xi64>
      %86 = stablehlo.compare EQ, %85, %83, SIGNED : (tensor<12x12xi64>, tensor<12x12xi64>) -> tensor<12x12xi1>
      %87 = stablehlo.convert %86 : (tensor<12x12xi1>) -> tensor<12x12xf64>
      %cst_64 = stablehlo.constant dense<1.000000e-10> : tensor<f64>
      %88 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f64>) -> tensor<12x12xf64>
      %89 = stablehlo.multiply %88, %87 : tensor<12x12xf64>
      %90 = stablehlo.broadcast_in_dim %89, dims = [1, 2] : (tensor<12x12xf64>) -> tensor<1x12x12xf64>
      %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<1x12x12xf64>) -> tensor<256x12x12xf64>
      %92 = stablehlo.add %81, %91 : tensor<256x12x12xf64>
      %93 = stablehlo.transpose %iterArg_31, dims = [1, 0] : (tensor<192x12xf64>) -> tensor<12x192xf64>
      %94 = stablehlo.multiply %37, %41 : tensor<256x192xf64>
      %95 = stablehlo.dot_general %93, %94, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<12x192xf64>, tensor<256x192xf64>) -> tensor<12x256xf64>
      %96 = stablehlo.slice %92 [0:256, 0:1, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %97 = stablehlo.reshape %96 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %98 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %99 = stablehlo.subtract %97, %98 : tensor<256xf64>
      %100 = stablehlo.sqrt %99 : tensor<256xf64>
      %101 = stablehlo.slice %92 [0:256, 1:2, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %102 = stablehlo.reshape %101 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %103 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %104 = stablehlo.subtract %102, %103 : tensor<256xf64>
      %105 = stablehlo.divide %104, %100 : tensor<256xf64>
      %106 = stablehlo.slice %92 [0:256, 2:3, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %107 = stablehlo.reshape %106 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %108 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %109 = stablehlo.subtract %107, %108 : tensor<256xf64>
      %110 = stablehlo.divide %109, %100 : tensor<256xf64>
      %111 = stablehlo.slice %92 [0:256, 3:4, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %112 = stablehlo.reshape %111 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %113 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %114 = stablehlo.subtract %112, %113 : tensor<256xf64>
      %115 = stablehlo.divide %114, %100 : tensor<256xf64>
      %116 = stablehlo.slice %92 [0:256, 4:5, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %117 = stablehlo.reshape %116 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %118 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %119 = stablehlo.subtract %117, %118 : tensor<256xf64>
      %120 = stablehlo.divide %119, %100 : tensor<256xf64>
      %121 = stablehlo.slice %92 [0:256, 5:6, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %122 = stablehlo.reshape %121 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_70 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %123 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %124 = stablehlo.subtract %122, %123 : tensor<256xf64>
      %125 = stablehlo.divide %124, %100 : tensor<256xf64>
      %126 = stablehlo.slice %92 [0:256, 6:7, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %127 = stablehlo.reshape %126 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %128 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %129 = stablehlo.subtract %127, %128 : tensor<256xf64>
      %130 = stablehlo.divide %129, %100 : tensor<256xf64>
      %131 = stablehlo.slice %92 [0:256, 7:8, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %132 = stablehlo.reshape %131 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_72 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %133 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %134 = stablehlo.subtract %132, %133 : tensor<256xf64>
      %135 = stablehlo.divide %134, %100 : tensor<256xf64>
      %136 = stablehlo.slice %92 [0:256, 8:9, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %137 = stablehlo.reshape %136 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %138 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %139 = stablehlo.subtract %137, %138 : tensor<256xf64>
      %140 = stablehlo.divide %139, %100 : tensor<256xf64>
      %141 = stablehlo.slice %92 [0:256, 9:10, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %142 = stablehlo.reshape %141 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %143 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %144 = stablehlo.subtract %142, %143 : tensor<256xf64>
      %145 = stablehlo.divide %144, %100 : tensor<256xf64>
      %146 = stablehlo.slice %92 [0:256, 10:11, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %147 = stablehlo.reshape %146 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %148 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %149 = stablehlo.subtract %147, %148 : tensor<256xf64>
      %150 = stablehlo.divide %149, %100 : tensor<256xf64>
      %151 = stablehlo.slice %92 [0:256, 11:12, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %152 = stablehlo.reshape %151 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %153 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %154 = stablehlo.subtract %152, %153 : tensor<256xf64>
      %155 = stablehlo.divide %154, %100 : tensor<256xf64>
      %156 = stablehlo.slice %92 [0:256, 1:2, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %157 = stablehlo.reshape %156 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %158 = stablehlo.multiply %105, %105 : tensor<256xf64>
      %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %159 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %160 = stablehlo.add %159, %158 : tensor<256xf64>
      %161 = stablehlo.subtract %157, %160 : tensor<256xf64>
      %162 = stablehlo.sqrt %161 : tensor<256xf64>
      %163 = stablehlo.slice %92 [0:256, 2:3, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %164 = stablehlo.reshape %163 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %165 = stablehlo.multiply %110, %105 : tensor<256xf64>
      %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %166 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %167 = stablehlo.add %166, %165 : tensor<256xf64>
      %168 = stablehlo.subtract %164, %167 : tensor<256xf64>
      %169 = stablehlo.divide %168, %162 : tensor<256xf64>
      %170 = stablehlo.slice %92 [0:256, 3:4, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %171 = stablehlo.reshape %170 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %172 = stablehlo.multiply %115, %105 : tensor<256xf64>
      %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %173 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %174 = stablehlo.add %173, %172 : tensor<256xf64>
      %175 = stablehlo.subtract %171, %174 : tensor<256xf64>
      %176 = stablehlo.divide %175, %162 : tensor<256xf64>
      %177 = stablehlo.slice %92 [0:256, 4:5, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %178 = stablehlo.reshape %177 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %179 = stablehlo.multiply %120, %105 : tensor<256xf64>
      %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %180 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %181 = stablehlo.add %180, %179 : tensor<256xf64>
      %182 = stablehlo.subtract %178, %181 : tensor<256xf64>
      %183 = stablehlo.divide %182, %162 : tensor<256xf64>
      %184 = stablehlo.slice %92 [0:256, 5:6, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %185 = stablehlo.reshape %184 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %186 = stablehlo.multiply %125, %105 : tensor<256xf64>
      %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %187 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %188 = stablehlo.add %187, %186 : tensor<256xf64>
      %189 = stablehlo.subtract %185, %188 : tensor<256xf64>
      %190 = stablehlo.divide %189, %162 : tensor<256xf64>
      %191 = stablehlo.slice %92 [0:256, 6:7, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %192 = stablehlo.reshape %191 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %193 = stablehlo.multiply %130, %105 : tensor<256xf64>
      %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %194 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %195 = stablehlo.add %194, %193 : tensor<256xf64>
      %196 = stablehlo.subtract %192, %195 : tensor<256xf64>
      %197 = stablehlo.divide %196, %162 : tensor<256xf64>
      %198 = stablehlo.slice %92 [0:256, 7:8, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %199 = stablehlo.reshape %198 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %200 = stablehlo.multiply %135, %105 : tensor<256xf64>
      %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %201 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %202 = stablehlo.add %201, %200 : tensor<256xf64>
      %203 = stablehlo.subtract %199, %202 : tensor<256xf64>
      %204 = stablehlo.divide %203, %162 : tensor<256xf64>
      %205 = stablehlo.slice %92 [0:256, 8:9, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %206 = stablehlo.reshape %205 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %207 = stablehlo.multiply %140, %105 : tensor<256xf64>
      %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %208 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %209 = stablehlo.add %208, %207 : tensor<256xf64>
      %210 = stablehlo.subtract %206, %209 : tensor<256xf64>
      %211 = stablehlo.divide %210, %162 : tensor<256xf64>
      %212 = stablehlo.slice %92 [0:256, 9:10, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %213 = stablehlo.reshape %212 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %214 = stablehlo.multiply %145, %105 : tensor<256xf64>
      %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %215 = stablehlo.broadcast_in_dim %cst_85, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %216 = stablehlo.add %215, %214 : tensor<256xf64>
      %217 = stablehlo.subtract %213, %216 : tensor<256xf64>
      %218 = stablehlo.divide %217, %162 : tensor<256xf64>
      %219 = stablehlo.slice %92 [0:256, 10:11, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %220 = stablehlo.reshape %219 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %221 = stablehlo.multiply %150, %105 : tensor<256xf64>
      %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %222 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %223 = stablehlo.add %222, %221 : tensor<256xf64>
      %224 = stablehlo.subtract %220, %223 : tensor<256xf64>
      %225 = stablehlo.divide %224, %162 : tensor<256xf64>
      %226 = stablehlo.slice %92 [0:256, 11:12, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %227 = stablehlo.reshape %226 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %228 = stablehlo.multiply %155, %105 : tensor<256xf64>
      %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %229 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %230 = stablehlo.add %229, %228 : tensor<256xf64>
      %231 = stablehlo.subtract %227, %230 : tensor<256xf64>
      %232 = stablehlo.divide %231, %162 : tensor<256xf64>
      %233 = stablehlo.slice %92 [0:256, 2:3, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %234 = stablehlo.reshape %233 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %235 = stablehlo.multiply %110, %110 : tensor<256xf64>
      %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %236 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %237 = stablehlo.add %236, %235 : tensor<256xf64>
      %238 = stablehlo.multiply %169, %169 : tensor<256xf64>
      %239 = stablehlo.add %237, %238 : tensor<256xf64>
      %240 = stablehlo.subtract %234, %239 : tensor<256xf64>
      %241 = stablehlo.sqrt %240 : tensor<256xf64>
      %242 = stablehlo.slice %92 [0:256, 3:4, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %243 = stablehlo.reshape %242 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %244 = stablehlo.multiply %115, %110 : tensor<256xf64>
      %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %245 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %246 = stablehlo.add %245, %244 : tensor<256xf64>
      %247 = stablehlo.multiply %176, %169 : tensor<256xf64>
      %248 = stablehlo.add %246, %247 : tensor<256xf64>
      %249 = stablehlo.subtract %243, %248 : tensor<256xf64>
      %250 = stablehlo.divide %249, %241 : tensor<256xf64>
      %251 = stablehlo.slice %92 [0:256, 4:5, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %252 = stablehlo.reshape %251 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %253 = stablehlo.multiply %120, %110 : tensor<256xf64>
      %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %254 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %255 = stablehlo.add %254, %253 : tensor<256xf64>
      %256 = stablehlo.multiply %183, %169 : tensor<256xf64>
      %257 = stablehlo.add %255, %256 : tensor<256xf64>
      %258 = stablehlo.subtract %252, %257 : tensor<256xf64>
      %259 = stablehlo.divide %258, %241 : tensor<256xf64>
      %260 = stablehlo.slice %92 [0:256, 5:6, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %261 = stablehlo.reshape %260 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %262 = stablehlo.multiply %125, %110 : tensor<256xf64>
      %cst_91 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %263 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %264 = stablehlo.add %263, %262 : tensor<256xf64>
      %265 = stablehlo.multiply %190, %169 : tensor<256xf64>
      %266 = stablehlo.add %264, %265 : tensor<256xf64>
      %267 = stablehlo.subtract %261, %266 : tensor<256xf64>
      %268 = stablehlo.divide %267, %241 : tensor<256xf64>
      %269 = stablehlo.slice %92 [0:256, 6:7, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %270 = stablehlo.reshape %269 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %271 = stablehlo.multiply %130, %110 : tensor<256xf64>
      %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %272 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %273 = stablehlo.add %272, %271 : tensor<256xf64>
      %274 = stablehlo.multiply %197, %169 : tensor<256xf64>
      %275 = stablehlo.add %273, %274 : tensor<256xf64>
      %276 = stablehlo.subtract %270, %275 : tensor<256xf64>
      %277 = stablehlo.divide %276, %241 : tensor<256xf64>
      %278 = stablehlo.slice %92 [0:256, 7:8, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %279 = stablehlo.reshape %278 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %280 = stablehlo.multiply %135, %110 : tensor<256xf64>
      %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %281 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %282 = stablehlo.add %281, %280 : tensor<256xf64>
      %283 = stablehlo.multiply %204, %169 : tensor<256xf64>
      %284 = stablehlo.add %282, %283 : tensor<256xf64>
      %285 = stablehlo.subtract %279, %284 : tensor<256xf64>
      %286 = stablehlo.divide %285, %241 : tensor<256xf64>
      %287 = stablehlo.slice %92 [0:256, 8:9, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %288 = stablehlo.reshape %287 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %289 = stablehlo.multiply %140, %110 : tensor<256xf64>
      %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %290 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %291 = stablehlo.add %290, %289 : tensor<256xf64>
      %292 = stablehlo.multiply %211, %169 : tensor<256xf64>
      %293 = stablehlo.add %291, %292 : tensor<256xf64>
      %294 = stablehlo.subtract %288, %293 : tensor<256xf64>
      %295 = stablehlo.divide %294, %241 : tensor<256xf64>
      %296 = stablehlo.slice %92 [0:256, 9:10, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %297 = stablehlo.reshape %296 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %298 = stablehlo.multiply %145, %110 : tensor<256xf64>
      %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %299 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %300 = stablehlo.add %299, %298 : tensor<256xf64>
      %301 = stablehlo.multiply %218, %169 : tensor<256xf64>
      %302 = stablehlo.add %300, %301 : tensor<256xf64>
      %303 = stablehlo.subtract %297, %302 : tensor<256xf64>
      %304 = stablehlo.divide %303, %241 : tensor<256xf64>
      %305 = stablehlo.slice %92 [0:256, 10:11, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %306 = stablehlo.reshape %305 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %307 = stablehlo.multiply %150, %110 : tensor<256xf64>
      %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %308 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %309 = stablehlo.add %308, %307 : tensor<256xf64>
      %310 = stablehlo.multiply %225, %169 : tensor<256xf64>
      %311 = stablehlo.add %309, %310 : tensor<256xf64>
      %312 = stablehlo.subtract %306, %311 : tensor<256xf64>
      %313 = stablehlo.divide %312, %241 : tensor<256xf64>
      %314 = stablehlo.slice %92 [0:256, 11:12, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %315 = stablehlo.reshape %314 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %316 = stablehlo.multiply %155, %110 : tensor<256xf64>
      %cst_97 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %317 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %318 = stablehlo.add %317, %316 : tensor<256xf64>
      %319 = stablehlo.multiply %232, %169 : tensor<256xf64>
      %320 = stablehlo.add %318, %319 : tensor<256xf64>
      %321 = stablehlo.subtract %315, %320 : tensor<256xf64>
      %322 = stablehlo.divide %321, %241 : tensor<256xf64>
      %323 = stablehlo.slice %92 [0:256, 3:4, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %324 = stablehlo.reshape %323 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %325 = stablehlo.multiply %115, %115 : tensor<256xf64>
      %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %326 = stablehlo.broadcast_in_dim %cst_98, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %327 = stablehlo.add %326, %325 : tensor<256xf64>
      %328 = stablehlo.multiply %176, %176 : tensor<256xf64>
      %329 = stablehlo.add %327, %328 : tensor<256xf64>
      %330 = stablehlo.multiply %250, %250 : tensor<256xf64>
      %331 = stablehlo.add %329, %330 : tensor<256xf64>
      %332 = stablehlo.subtract %324, %331 : tensor<256xf64>
      %333 = stablehlo.sqrt %332 : tensor<256xf64>
      %334 = stablehlo.slice %92 [0:256, 4:5, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %335 = stablehlo.reshape %334 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %336 = stablehlo.multiply %120, %115 : tensor<256xf64>
      %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %337 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %338 = stablehlo.add %337, %336 : tensor<256xf64>
      %339 = stablehlo.multiply %183, %176 : tensor<256xf64>
      %340 = stablehlo.add %338, %339 : tensor<256xf64>
      %341 = stablehlo.multiply %259, %250 : tensor<256xf64>
      %342 = stablehlo.add %340, %341 : tensor<256xf64>
      %343 = stablehlo.subtract %335, %342 : tensor<256xf64>
      %344 = stablehlo.divide %343, %333 : tensor<256xf64>
      %345 = stablehlo.slice %92 [0:256, 5:6, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %346 = stablehlo.reshape %345 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %347 = stablehlo.multiply %125, %115 : tensor<256xf64>
      %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %348 = stablehlo.broadcast_in_dim %cst_100, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %349 = stablehlo.add %348, %347 : tensor<256xf64>
      %350 = stablehlo.multiply %190, %176 : tensor<256xf64>
      %351 = stablehlo.add %349, %350 : tensor<256xf64>
      %352 = stablehlo.multiply %268, %250 : tensor<256xf64>
      %353 = stablehlo.add %351, %352 : tensor<256xf64>
      %354 = stablehlo.subtract %346, %353 : tensor<256xf64>
      %355 = stablehlo.divide %354, %333 : tensor<256xf64>
      %356 = stablehlo.slice %92 [0:256, 6:7, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %357 = stablehlo.reshape %356 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %358 = stablehlo.multiply %130, %115 : tensor<256xf64>
      %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %359 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %360 = stablehlo.add %359, %358 : tensor<256xf64>
      %361 = stablehlo.multiply %197, %176 : tensor<256xf64>
      %362 = stablehlo.add %360, %361 : tensor<256xf64>
      %363 = stablehlo.multiply %277, %250 : tensor<256xf64>
      %364 = stablehlo.add %362, %363 : tensor<256xf64>
      %365 = stablehlo.subtract %357, %364 : tensor<256xf64>
      %366 = stablehlo.divide %365, %333 : tensor<256xf64>
      %367 = stablehlo.slice %92 [0:256, 7:8, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %368 = stablehlo.reshape %367 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %369 = stablehlo.multiply %135, %115 : tensor<256xf64>
      %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %370 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %371 = stablehlo.add %370, %369 : tensor<256xf64>
      %372 = stablehlo.multiply %204, %176 : tensor<256xf64>
      %373 = stablehlo.add %371, %372 : tensor<256xf64>
      %374 = stablehlo.multiply %286, %250 : tensor<256xf64>
      %375 = stablehlo.add %373, %374 : tensor<256xf64>
      %376 = stablehlo.subtract %368, %375 : tensor<256xf64>
      %377 = stablehlo.divide %376, %333 : tensor<256xf64>
      %378 = stablehlo.slice %92 [0:256, 8:9, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %379 = stablehlo.reshape %378 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %380 = stablehlo.multiply %140, %115 : tensor<256xf64>
      %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %381 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %382 = stablehlo.add %381, %380 : tensor<256xf64>
      %383 = stablehlo.multiply %211, %176 : tensor<256xf64>
      %384 = stablehlo.add %382, %383 : tensor<256xf64>
      %385 = stablehlo.multiply %295, %250 : tensor<256xf64>
      %386 = stablehlo.add %384, %385 : tensor<256xf64>
      %387 = stablehlo.subtract %379, %386 : tensor<256xf64>
      %388 = stablehlo.divide %387, %333 : tensor<256xf64>
      %389 = stablehlo.slice %92 [0:256, 9:10, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %390 = stablehlo.reshape %389 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %391 = stablehlo.multiply %145, %115 : tensor<256xf64>
      %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %392 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %393 = stablehlo.add %392, %391 : tensor<256xf64>
      %394 = stablehlo.multiply %218, %176 : tensor<256xf64>
      %395 = stablehlo.add %393, %394 : tensor<256xf64>
      %396 = stablehlo.multiply %304, %250 : tensor<256xf64>
      %397 = stablehlo.add %395, %396 : tensor<256xf64>
      %398 = stablehlo.subtract %390, %397 : tensor<256xf64>
      %399 = stablehlo.divide %398, %333 : tensor<256xf64>
      %400 = stablehlo.slice %92 [0:256, 10:11, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %401 = stablehlo.reshape %400 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %402 = stablehlo.multiply %150, %115 : tensor<256xf64>
      %cst_105 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %403 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %404 = stablehlo.add %403, %402 : tensor<256xf64>
      %405 = stablehlo.multiply %225, %176 : tensor<256xf64>
      %406 = stablehlo.add %404, %405 : tensor<256xf64>
      %407 = stablehlo.multiply %313, %250 : tensor<256xf64>
      %408 = stablehlo.add %406, %407 : tensor<256xf64>
      %409 = stablehlo.subtract %401, %408 : tensor<256xf64>
      %410 = stablehlo.divide %409, %333 : tensor<256xf64>
      %411 = stablehlo.slice %92 [0:256, 11:12, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %412 = stablehlo.reshape %411 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %413 = stablehlo.multiply %155, %115 : tensor<256xf64>
      %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %414 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %415 = stablehlo.add %414, %413 : tensor<256xf64>
      %416 = stablehlo.multiply %232, %176 : tensor<256xf64>
      %417 = stablehlo.add %415, %416 : tensor<256xf64>
      %418 = stablehlo.multiply %322, %250 : tensor<256xf64>
      %419 = stablehlo.add %417, %418 : tensor<256xf64>
      %420 = stablehlo.subtract %412, %419 : tensor<256xf64>
      %421 = stablehlo.divide %420, %333 : tensor<256xf64>
      %422 = stablehlo.slice %92 [0:256, 4:5, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %423 = stablehlo.reshape %422 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %424 = stablehlo.multiply %120, %120 : tensor<256xf64>
      %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %425 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %426 = stablehlo.add %425, %424 : tensor<256xf64>
      %427 = stablehlo.multiply %183, %183 : tensor<256xf64>
      %428 = stablehlo.add %426, %427 : tensor<256xf64>
      %429 = stablehlo.multiply %259, %259 : tensor<256xf64>
      %430 = stablehlo.add %428, %429 : tensor<256xf64>
      %431 = stablehlo.multiply %344, %344 : tensor<256xf64>
      %432 = stablehlo.add %430, %431 : tensor<256xf64>
      %433 = stablehlo.subtract %423, %432 : tensor<256xf64>
      %434 = stablehlo.sqrt %433 : tensor<256xf64>
      %435 = stablehlo.slice %92 [0:256, 5:6, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %436 = stablehlo.reshape %435 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %437 = stablehlo.multiply %125, %120 : tensor<256xf64>
      %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %438 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %439 = stablehlo.add %438, %437 : tensor<256xf64>
      %440 = stablehlo.multiply %190, %183 : tensor<256xf64>
      %441 = stablehlo.add %439, %440 : tensor<256xf64>
      %442 = stablehlo.multiply %268, %259 : tensor<256xf64>
      %443 = stablehlo.add %441, %442 : tensor<256xf64>
      %444 = stablehlo.multiply %355, %344 : tensor<256xf64>
      %445 = stablehlo.add %443, %444 : tensor<256xf64>
      %446 = stablehlo.subtract %436, %445 : tensor<256xf64>
      %447 = stablehlo.divide %446, %434 : tensor<256xf64>
      %448 = stablehlo.slice %92 [0:256, 6:7, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %449 = stablehlo.reshape %448 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %450 = stablehlo.multiply %130, %120 : tensor<256xf64>
      %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %451 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %452 = stablehlo.add %451, %450 : tensor<256xf64>
      %453 = stablehlo.multiply %197, %183 : tensor<256xf64>
      %454 = stablehlo.add %452, %453 : tensor<256xf64>
      %455 = stablehlo.multiply %277, %259 : tensor<256xf64>
      %456 = stablehlo.add %454, %455 : tensor<256xf64>
      %457 = stablehlo.multiply %366, %344 : tensor<256xf64>
      %458 = stablehlo.add %456, %457 : tensor<256xf64>
      %459 = stablehlo.subtract %449, %458 : tensor<256xf64>
      %460 = stablehlo.divide %459, %434 : tensor<256xf64>
      %461 = stablehlo.slice %92 [0:256, 7:8, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %462 = stablehlo.reshape %461 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %463 = stablehlo.multiply %135, %120 : tensor<256xf64>
      %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %464 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %465 = stablehlo.add %464, %463 : tensor<256xf64>
      %466 = stablehlo.multiply %204, %183 : tensor<256xf64>
      %467 = stablehlo.add %465, %466 : tensor<256xf64>
      %468 = stablehlo.multiply %286, %259 : tensor<256xf64>
      %469 = stablehlo.add %467, %468 : tensor<256xf64>
      %470 = stablehlo.multiply %377, %344 : tensor<256xf64>
      %471 = stablehlo.add %469, %470 : tensor<256xf64>
      %472 = stablehlo.subtract %462, %471 : tensor<256xf64>
      %473 = stablehlo.divide %472, %434 : tensor<256xf64>
      %474 = stablehlo.slice %92 [0:256, 8:9, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %475 = stablehlo.reshape %474 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %476 = stablehlo.multiply %140, %120 : tensor<256xf64>
      %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %477 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %478 = stablehlo.add %477, %476 : tensor<256xf64>
      %479 = stablehlo.multiply %211, %183 : tensor<256xf64>
      %480 = stablehlo.add %478, %479 : tensor<256xf64>
      %481 = stablehlo.multiply %295, %259 : tensor<256xf64>
      %482 = stablehlo.add %480, %481 : tensor<256xf64>
      %483 = stablehlo.multiply %388, %344 : tensor<256xf64>
      %484 = stablehlo.add %482, %483 : tensor<256xf64>
      %485 = stablehlo.subtract %475, %484 : tensor<256xf64>
      %486 = stablehlo.divide %485, %434 : tensor<256xf64>
      %487 = stablehlo.slice %92 [0:256, 9:10, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %488 = stablehlo.reshape %487 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %489 = stablehlo.multiply %145, %120 : tensor<256xf64>
      %cst_112 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %490 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %491 = stablehlo.add %490, %489 : tensor<256xf64>
      %492 = stablehlo.multiply %218, %183 : tensor<256xf64>
      %493 = stablehlo.add %491, %492 : tensor<256xf64>
      %494 = stablehlo.multiply %304, %259 : tensor<256xf64>
      %495 = stablehlo.add %493, %494 : tensor<256xf64>
      %496 = stablehlo.multiply %399, %344 : tensor<256xf64>
      %497 = stablehlo.add %495, %496 : tensor<256xf64>
      %498 = stablehlo.subtract %488, %497 : tensor<256xf64>
      %499 = stablehlo.divide %498, %434 : tensor<256xf64>
      %500 = stablehlo.slice %92 [0:256, 10:11, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %501 = stablehlo.reshape %500 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %502 = stablehlo.multiply %150, %120 : tensor<256xf64>
      %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %503 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %504 = stablehlo.add %503, %502 : tensor<256xf64>
      %505 = stablehlo.multiply %225, %183 : tensor<256xf64>
      %506 = stablehlo.add %504, %505 : tensor<256xf64>
      %507 = stablehlo.multiply %313, %259 : tensor<256xf64>
      %508 = stablehlo.add %506, %507 : tensor<256xf64>
      %509 = stablehlo.multiply %410, %344 : tensor<256xf64>
      %510 = stablehlo.add %508, %509 : tensor<256xf64>
      %511 = stablehlo.subtract %501, %510 : tensor<256xf64>
      %512 = stablehlo.divide %511, %434 : tensor<256xf64>
      %513 = stablehlo.slice %92 [0:256, 11:12, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %514 = stablehlo.reshape %513 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %515 = stablehlo.multiply %155, %120 : tensor<256xf64>
      %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %516 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %517 = stablehlo.add %516, %515 : tensor<256xf64>
      %518 = stablehlo.multiply %232, %183 : tensor<256xf64>
      %519 = stablehlo.add %517, %518 : tensor<256xf64>
      %520 = stablehlo.multiply %322, %259 : tensor<256xf64>
      %521 = stablehlo.add %519, %520 : tensor<256xf64>
      %522 = stablehlo.multiply %421, %344 : tensor<256xf64>
      %523 = stablehlo.add %521, %522 : tensor<256xf64>
      %524 = stablehlo.subtract %514, %523 : tensor<256xf64>
      %525 = stablehlo.divide %524, %434 : tensor<256xf64>
      %526 = stablehlo.slice %92 [0:256, 5:6, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %527 = stablehlo.reshape %526 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %528 = stablehlo.multiply %125, %125 : tensor<256xf64>
      %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %529 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %530 = stablehlo.add %529, %528 : tensor<256xf64>
      %531 = stablehlo.multiply %190, %190 : tensor<256xf64>
      %532 = stablehlo.add %530, %531 : tensor<256xf64>
      %533 = stablehlo.multiply %268, %268 : tensor<256xf64>
      %534 = stablehlo.add %532, %533 : tensor<256xf64>
      %535 = stablehlo.multiply %355, %355 : tensor<256xf64>
      %536 = stablehlo.add %534, %535 : tensor<256xf64>
      %537 = stablehlo.multiply %447, %447 : tensor<256xf64>
      %538 = stablehlo.add %536, %537 : tensor<256xf64>
      %539 = stablehlo.subtract %527, %538 : tensor<256xf64>
      %540 = stablehlo.sqrt %539 : tensor<256xf64>
      %541 = stablehlo.slice %92 [0:256, 6:7, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %542 = stablehlo.reshape %541 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %543 = stablehlo.multiply %130, %125 : tensor<256xf64>
      %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %544 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %545 = stablehlo.add %544, %543 : tensor<256xf64>
      %546 = stablehlo.multiply %197, %190 : tensor<256xf64>
      %547 = stablehlo.add %545, %546 : tensor<256xf64>
      %548 = stablehlo.multiply %277, %268 : tensor<256xf64>
      %549 = stablehlo.add %547, %548 : tensor<256xf64>
      %550 = stablehlo.multiply %366, %355 : tensor<256xf64>
      %551 = stablehlo.add %549, %550 : tensor<256xf64>
      %552 = stablehlo.multiply %460, %447 : tensor<256xf64>
      %553 = stablehlo.add %551, %552 : tensor<256xf64>
      %554 = stablehlo.subtract %542, %553 : tensor<256xf64>
      %555 = stablehlo.divide %554, %540 : tensor<256xf64>
      %556 = stablehlo.slice %92 [0:256, 7:8, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %557 = stablehlo.reshape %556 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %558 = stablehlo.multiply %135, %125 : tensor<256xf64>
      %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %559 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %560 = stablehlo.add %559, %558 : tensor<256xf64>
      %561 = stablehlo.multiply %204, %190 : tensor<256xf64>
      %562 = stablehlo.add %560, %561 : tensor<256xf64>
      %563 = stablehlo.multiply %286, %268 : tensor<256xf64>
      %564 = stablehlo.add %562, %563 : tensor<256xf64>
      %565 = stablehlo.multiply %377, %355 : tensor<256xf64>
      %566 = stablehlo.add %564, %565 : tensor<256xf64>
      %567 = stablehlo.multiply %473, %447 : tensor<256xf64>
      %568 = stablehlo.add %566, %567 : tensor<256xf64>
      %569 = stablehlo.subtract %557, %568 : tensor<256xf64>
      %570 = stablehlo.divide %569, %540 : tensor<256xf64>
      %571 = stablehlo.slice %92 [0:256, 8:9, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %572 = stablehlo.reshape %571 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %573 = stablehlo.multiply %140, %125 : tensor<256xf64>
      %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %574 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %575 = stablehlo.add %574, %573 : tensor<256xf64>
      %576 = stablehlo.multiply %211, %190 : tensor<256xf64>
      %577 = stablehlo.add %575, %576 : tensor<256xf64>
      %578 = stablehlo.multiply %295, %268 : tensor<256xf64>
      %579 = stablehlo.add %577, %578 : tensor<256xf64>
      %580 = stablehlo.multiply %388, %355 : tensor<256xf64>
      %581 = stablehlo.add %579, %580 : tensor<256xf64>
      %582 = stablehlo.multiply %486, %447 : tensor<256xf64>
      %583 = stablehlo.add %581, %582 : tensor<256xf64>
      %584 = stablehlo.subtract %572, %583 : tensor<256xf64>
      %585 = stablehlo.divide %584, %540 : tensor<256xf64>
      %586 = stablehlo.slice %92 [0:256, 9:10, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %587 = stablehlo.reshape %586 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %588 = stablehlo.multiply %145, %125 : tensor<256xf64>
      %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %589 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %590 = stablehlo.add %589, %588 : tensor<256xf64>
      %591 = stablehlo.multiply %218, %190 : tensor<256xf64>
      %592 = stablehlo.add %590, %591 : tensor<256xf64>
      %593 = stablehlo.multiply %304, %268 : tensor<256xf64>
      %594 = stablehlo.add %592, %593 : tensor<256xf64>
      %595 = stablehlo.multiply %399, %355 : tensor<256xf64>
      %596 = stablehlo.add %594, %595 : tensor<256xf64>
      %597 = stablehlo.multiply %499, %447 : tensor<256xf64>
      %598 = stablehlo.add %596, %597 : tensor<256xf64>
      %599 = stablehlo.subtract %587, %598 : tensor<256xf64>
      %600 = stablehlo.divide %599, %540 : tensor<256xf64>
      %601 = stablehlo.slice %92 [0:256, 10:11, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %602 = stablehlo.reshape %601 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %603 = stablehlo.multiply %150, %125 : tensor<256xf64>
      %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %604 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %605 = stablehlo.add %604, %603 : tensor<256xf64>
      %606 = stablehlo.multiply %225, %190 : tensor<256xf64>
      %607 = stablehlo.add %605, %606 : tensor<256xf64>
      %608 = stablehlo.multiply %313, %268 : tensor<256xf64>
      %609 = stablehlo.add %607, %608 : tensor<256xf64>
      %610 = stablehlo.multiply %410, %355 : tensor<256xf64>
      %611 = stablehlo.add %609, %610 : tensor<256xf64>
      %612 = stablehlo.multiply %512, %447 : tensor<256xf64>
      %613 = stablehlo.add %611, %612 : tensor<256xf64>
      %614 = stablehlo.subtract %602, %613 : tensor<256xf64>
      %615 = stablehlo.divide %614, %540 : tensor<256xf64>
      %616 = stablehlo.slice %92 [0:256, 11:12, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %617 = stablehlo.reshape %616 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %618 = stablehlo.multiply %155, %125 : tensor<256xf64>
      %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %619 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %620 = stablehlo.add %619, %618 : tensor<256xf64>
      %621 = stablehlo.multiply %232, %190 : tensor<256xf64>
      %622 = stablehlo.add %620, %621 : tensor<256xf64>
      %623 = stablehlo.multiply %322, %268 : tensor<256xf64>
      %624 = stablehlo.add %622, %623 : tensor<256xf64>
      %625 = stablehlo.multiply %421, %355 : tensor<256xf64>
      %626 = stablehlo.add %624, %625 : tensor<256xf64>
      %627 = stablehlo.multiply %525, %447 : tensor<256xf64>
      %628 = stablehlo.add %626, %627 : tensor<256xf64>
      %629 = stablehlo.subtract %617, %628 : tensor<256xf64>
      %630 = stablehlo.divide %629, %540 : tensor<256xf64>
      %631 = stablehlo.slice %92 [0:256, 6:7, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %632 = stablehlo.reshape %631 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %633 = stablehlo.multiply %130, %130 : tensor<256xf64>
      %cst_122 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %634 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %635 = stablehlo.add %634, %633 : tensor<256xf64>
      %636 = stablehlo.multiply %197, %197 : tensor<256xf64>
      %637 = stablehlo.add %635, %636 : tensor<256xf64>
      %638 = stablehlo.multiply %277, %277 : tensor<256xf64>
      %639 = stablehlo.add %637, %638 : tensor<256xf64>
      %640 = stablehlo.multiply %366, %366 : tensor<256xf64>
      %641 = stablehlo.add %639, %640 : tensor<256xf64>
      %642 = stablehlo.multiply %460, %460 : tensor<256xf64>
      %643 = stablehlo.add %641, %642 : tensor<256xf64>
      %644 = stablehlo.multiply %555, %555 : tensor<256xf64>
      %645 = stablehlo.add %643, %644 : tensor<256xf64>
      %646 = stablehlo.subtract %632, %645 : tensor<256xf64>
      %647 = stablehlo.sqrt %646 : tensor<256xf64>
      %648 = stablehlo.slice %92 [0:256, 7:8, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %649 = stablehlo.reshape %648 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %650 = stablehlo.multiply %135, %130 : tensor<256xf64>
      %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %651 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %652 = stablehlo.add %651, %650 : tensor<256xf64>
      %653 = stablehlo.multiply %204, %197 : tensor<256xf64>
      %654 = stablehlo.add %652, %653 : tensor<256xf64>
      %655 = stablehlo.multiply %286, %277 : tensor<256xf64>
      %656 = stablehlo.add %654, %655 : tensor<256xf64>
      %657 = stablehlo.multiply %377, %366 : tensor<256xf64>
      %658 = stablehlo.add %656, %657 : tensor<256xf64>
      %659 = stablehlo.multiply %473, %460 : tensor<256xf64>
      %660 = stablehlo.add %658, %659 : tensor<256xf64>
      %661 = stablehlo.multiply %570, %555 : tensor<256xf64>
      %662 = stablehlo.add %660, %661 : tensor<256xf64>
      %663 = stablehlo.subtract %649, %662 : tensor<256xf64>
      %664 = stablehlo.divide %663, %647 : tensor<256xf64>
      %665 = stablehlo.slice %92 [0:256, 8:9, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %666 = stablehlo.reshape %665 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %667 = stablehlo.multiply %140, %130 : tensor<256xf64>
      %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %668 = stablehlo.broadcast_in_dim %cst_124, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %669 = stablehlo.add %668, %667 : tensor<256xf64>
      %670 = stablehlo.multiply %211, %197 : tensor<256xf64>
      %671 = stablehlo.add %669, %670 : tensor<256xf64>
      %672 = stablehlo.multiply %295, %277 : tensor<256xf64>
      %673 = stablehlo.add %671, %672 : tensor<256xf64>
      %674 = stablehlo.multiply %388, %366 : tensor<256xf64>
      %675 = stablehlo.add %673, %674 : tensor<256xf64>
      %676 = stablehlo.multiply %486, %460 : tensor<256xf64>
      %677 = stablehlo.add %675, %676 : tensor<256xf64>
      %678 = stablehlo.multiply %585, %555 : tensor<256xf64>
      %679 = stablehlo.add %677, %678 : tensor<256xf64>
      %680 = stablehlo.subtract %666, %679 : tensor<256xf64>
      %681 = stablehlo.divide %680, %647 : tensor<256xf64>
      %682 = stablehlo.slice %92 [0:256, 9:10, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %683 = stablehlo.reshape %682 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %684 = stablehlo.multiply %145, %130 : tensor<256xf64>
      %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %685 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %686 = stablehlo.add %685, %684 : tensor<256xf64>
      %687 = stablehlo.multiply %218, %197 : tensor<256xf64>
      %688 = stablehlo.add %686, %687 : tensor<256xf64>
      %689 = stablehlo.multiply %304, %277 : tensor<256xf64>
      %690 = stablehlo.add %688, %689 : tensor<256xf64>
      %691 = stablehlo.multiply %399, %366 : tensor<256xf64>
      %692 = stablehlo.add %690, %691 : tensor<256xf64>
      %693 = stablehlo.multiply %499, %460 : tensor<256xf64>
      %694 = stablehlo.add %692, %693 : tensor<256xf64>
      %695 = stablehlo.multiply %600, %555 : tensor<256xf64>
      %696 = stablehlo.add %694, %695 : tensor<256xf64>
      %697 = stablehlo.subtract %683, %696 : tensor<256xf64>
      %698 = stablehlo.divide %697, %647 : tensor<256xf64>
      %699 = stablehlo.slice %92 [0:256, 10:11, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %700 = stablehlo.reshape %699 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %701 = stablehlo.multiply %150, %130 : tensor<256xf64>
      %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %702 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %703 = stablehlo.add %702, %701 : tensor<256xf64>
      %704 = stablehlo.multiply %225, %197 : tensor<256xf64>
      %705 = stablehlo.add %703, %704 : tensor<256xf64>
      %706 = stablehlo.multiply %313, %277 : tensor<256xf64>
      %707 = stablehlo.add %705, %706 : tensor<256xf64>
      %708 = stablehlo.multiply %410, %366 : tensor<256xf64>
      %709 = stablehlo.add %707, %708 : tensor<256xf64>
      %710 = stablehlo.multiply %512, %460 : tensor<256xf64>
      %711 = stablehlo.add %709, %710 : tensor<256xf64>
      %712 = stablehlo.multiply %615, %555 : tensor<256xf64>
      %713 = stablehlo.add %711, %712 : tensor<256xf64>
      %714 = stablehlo.subtract %700, %713 : tensor<256xf64>
      %715 = stablehlo.divide %714, %647 : tensor<256xf64>
      %716 = stablehlo.slice %92 [0:256, 11:12, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %717 = stablehlo.reshape %716 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %718 = stablehlo.multiply %155, %130 : tensor<256xf64>
      %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %719 = stablehlo.broadcast_in_dim %cst_127, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %720 = stablehlo.add %719, %718 : tensor<256xf64>
      %721 = stablehlo.multiply %232, %197 : tensor<256xf64>
      %722 = stablehlo.add %720, %721 : tensor<256xf64>
      %723 = stablehlo.multiply %322, %277 : tensor<256xf64>
      %724 = stablehlo.add %722, %723 : tensor<256xf64>
      %725 = stablehlo.multiply %421, %366 : tensor<256xf64>
      %726 = stablehlo.add %724, %725 : tensor<256xf64>
      %727 = stablehlo.multiply %525, %460 : tensor<256xf64>
      %728 = stablehlo.add %726, %727 : tensor<256xf64>
      %729 = stablehlo.multiply %630, %555 : tensor<256xf64>
      %730 = stablehlo.add %728, %729 : tensor<256xf64>
      %731 = stablehlo.subtract %717, %730 : tensor<256xf64>
      %732 = stablehlo.divide %731, %647 : tensor<256xf64>
      %733 = stablehlo.slice %92 [0:256, 7:8, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %734 = stablehlo.reshape %733 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %735 = stablehlo.multiply %135, %135 : tensor<256xf64>
      %cst_128 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %736 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %737 = stablehlo.add %736, %735 : tensor<256xf64>
      %738 = stablehlo.multiply %204, %204 : tensor<256xf64>
      %739 = stablehlo.add %737, %738 : tensor<256xf64>
      %740 = stablehlo.multiply %286, %286 : tensor<256xf64>
      %741 = stablehlo.add %739, %740 : tensor<256xf64>
      %742 = stablehlo.multiply %377, %377 : tensor<256xf64>
      %743 = stablehlo.add %741, %742 : tensor<256xf64>
      %744 = stablehlo.multiply %473, %473 : tensor<256xf64>
      %745 = stablehlo.add %743, %744 : tensor<256xf64>
      %746 = stablehlo.multiply %570, %570 : tensor<256xf64>
      %747 = stablehlo.add %745, %746 : tensor<256xf64>
      %748 = stablehlo.multiply %664, %664 : tensor<256xf64>
      %749 = stablehlo.add %747, %748 : tensor<256xf64>
      %750 = stablehlo.subtract %734, %749 : tensor<256xf64>
      %751 = stablehlo.sqrt %750 : tensor<256xf64>
      %752 = stablehlo.slice %92 [0:256, 8:9, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %753 = stablehlo.reshape %752 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %754 = stablehlo.multiply %140, %135 : tensor<256xf64>
      %cst_129 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %755 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %756 = stablehlo.add %755, %754 : tensor<256xf64>
      %757 = stablehlo.multiply %211, %204 : tensor<256xf64>
      %758 = stablehlo.add %756, %757 : tensor<256xf64>
      %759 = stablehlo.multiply %295, %286 : tensor<256xf64>
      %760 = stablehlo.add %758, %759 : tensor<256xf64>
      %761 = stablehlo.multiply %388, %377 : tensor<256xf64>
      %762 = stablehlo.add %760, %761 : tensor<256xf64>
      %763 = stablehlo.multiply %486, %473 : tensor<256xf64>
      %764 = stablehlo.add %762, %763 : tensor<256xf64>
      %765 = stablehlo.multiply %585, %570 : tensor<256xf64>
      %766 = stablehlo.add %764, %765 : tensor<256xf64>
      %767 = stablehlo.multiply %681, %664 : tensor<256xf64>
      %768 = stablehlo.add %766, %767 : tensor<256xf64>
      %769 = stablehlo.subtract %753, %768 : tensor<256xf64>
      %770 = stablehlo.divide %769, %751 : tensor<256xf64>
      %771 = stablehlo.slice %92 [0:256, 9:10, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %772 = stablehlo.reshape %771 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %773 = stablehlo.multiply %145, %135 : tensor<256xf64>
      %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %774 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %775 = stablehlo.add %774, %773 : tensor<256xf64>
      %776 = stablehlo.multiply %218, %204 : tensor<256xf64>
      %777 = stablehlo.add %775, %776 : tensor<256xf64>
      %778 = stablehlo.multiply %304, %286 : tensor<256xf64>
      %779 = stablehlo.add %777, %778 : tensor<256xf64>
      %780 = stablehlo.multiply %399, %377 : tensor<256xf64>
      %781 = stablehlo.add %779, %780 : tensor<256xf64>
      %782 = stablehlo.multiply %499, %473 : tensor<256xf64>
      %783 = stablehlo.add %781, %782 : tensor<256xf64>
      %784 = stablehlo.multiply %600, %570 : tensor<256xf64>
      %785 = stablehlo.add %783, %784 : tensor<256xf64>
      %786 = stablehlo.multiply %698, %664 : tensor<256xf64>
      %787 = stablehlo.add %785, %786 : tensor<256xf64>
      %788 = stablehlo.subtract %772, %787 : tensor<256xf64>
      %789 = stablehlo.divide %788, %751 : tensor<256xf64>
      %790 = stablehlo.slice %92 [0:256, 10:11, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %791 = stablehlo.reshape %790 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %792 = stablehlo.multiply %150, %135 : tensor<256xf64>
      %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %793 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %794 = stablehlo.add %793, %792 : tensor<256xf64>
      %795 = stablehlo.multiply %225, %204 : tensor<256xf64>
      %796 = stablehlo.add %794, %795 : tensor<256xf64>
      %797 = stablehlo.multiply %313, %286 : tensor<256xf64>
      %798 = stablehlo.add %796, %797 : tensor<256xf64>
      %799 = stablehlo.multiply %410, %377 : tensor<256xf64>
      %800 = stablehlo.add %798, %799 : tensor<256xf64>
      %801 = stablehlo.multiply %512, %473 : tensor<256xf64>
      %802 = stablehlo.add %800, %801 : tensor<256xf64>
      %803 = stablehlo.multiply %615, %570 : tensor<256xf64>
      %804 = stablehlo.add %802, %803 : tensor<256xf64>
      %805 = stablehlo.multiply %715, %664 : tensor<256xf64>
      %806 = stablehlo.add %804, %805 : tensor<256xf64>
      %807 = stablehlo.subtract %791, %806 : tensor<256xf64>
      %808 = stablehlo.divide %807, %751 : tensor<256xf64>
      %809 = stablehlo.slice %92 [0:256, 11:12, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %810 = stablehlo.reshape %809 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %811 = stablehlo.multiply %155, %135 : tensor<256xf64>
      %cst_132 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %812 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %813 = stablehlo.add %812, %811 : tensor<256xf64>
      %814 = stablehlo.multiply %232, %204 : tensor<256xf64>
      %815 = stablehlo.add %813, %814 : tensor<256xf64>
      %816 = stablehlo.multiply %322, %286 : tensor<256xf64>
      %817 = stablehlo.add %815, %816 : tensor<256xf64>
      %818 = stablehlo.multiply %421, %377 : tensor<256xf64>
      %819 = stablehlo.add %817, %818 : tensor<256xf64>
      %820 = stablehlo.multiply %525, %473 : tensor<256xf64>
      %821 = stablehlo.add %819, %820 : tensor<256xf64>
      %822 = stablehlo.multiply %630, %570 : tensor<256xf64>
      %823 = stablehlo.add %821, %822 : tensor<256xf64>
      %824 = stablehlo.multiply %732, %664 : tensor<256xf64>
      %825 = stablehlo.add %823, %824 : tensor<256xf64>
      %826 = stablehlo.subtract %810, %825 : tensor<256xf64>
      %827 = stablehlo.divide %826, %751 : tensor<256xf64>
      %828 = stablehlo.slice %92 [0:256, 8:9, 8:9] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %829 = stablehlo.reshape %828 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %830 = stablehlo.multiply %140, %140 : tensor<256xf64>
      %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %831 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %832 = stablehlo.add %831, %830 : tensor<256xf64>
      %833 = stablehlo.multiply %211, %211 : tensor<256xf64>
      %834 = stablehlo.add %832, %833 : tensor<256xf64>
      %835 = stablehlo.multiply %295, %295 : tensor<256xf64>
      %836 = stablehlo.add %834, %835 : tensor<256xf64>
      %837 = stablehlo.multiply %388, %388 : tensor<256xf64>
      %838 = stablehlo.add %836, %837 : tensor<256xf64>
      %839 = stablehlo.multiply %486, %486 : tensor<256xf64>
      %840 = stablehlo.add %838, %839 : tensor<256xf64>
      %841 = stablehlo.multiply %585, %585 : tensor<256xf64>
      %842 = stablehlo.add %840, %841 : tensor<256xf64>
      %843 = stablehlo.multiply %681, %681 : tensor<256xf64>
      %844 = stablehlo.add %842, %843 : tensor<256xf64>
      %845 = stablehlo.multiply %770, %770 : tensor<256xf64>
      %846 = stablehlo.add %844, %845 : tensor<256xf64>
      %847 = stablehlo.subtract %829, %846 : tensor<256xf64>
      %848 = stablehlo.sqrt %847 : tensor<256xf64>
      %849 = stablehlo.slice %92 [0:256, 9:10, 8:9] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %850 = stablehlo.reshape %849 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %851 = stablehlo.multiply %145, %140 : tensor<256xf64>
      %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %852 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %853 = stablehlo.add %852, %851 : tensor<256xf64>
      %854 = stablehlo.multiply %218, %211 : tensor<256xf64>
      %855 = stablehlo.add %853, %854 : tensor<256xf64>
      %856 = stablehlo.multiply %304, %295 : tensor<256xf64>
      %857 = stablehlo.add %855, %856 : tensor<256xf64>
      %858 = stablehlo.multiply %399, %388 : tensor<256xf64>
      %859 = stablehlo.add %857, %858 : tensor<256xf64>
      %860 = stablehlo.multiply %499, %486 : tensor<256xf64>
      %861 = stablehlo.add %859, %860 : tensor<256xf64>
      %862 = stablehlo.multiply %600, %585 : tensor<256xf64>
      %863 = stablehlo.add %861, %862 : tensor<256xf64>
      %864 = stablehlo.multiply %698, %681 : tensor<256xf64>
      %865 = stablehlo.add %863, %864 : tensor<256xf64>
      %866 = stablehlo.multiply %789, %770 : tensor<256xf64>
      %867 = stablehlo.add %865, %866 : tensor<256xf64>
      %868 = stablehlo.subtract %850, %867 : tensor<256xf64>
      %869 = stablehlo.divide %868, %848 : tensor<256xf64>
      %870 = stablehlo.slice %92 [0:256, 10:11, 8:9] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %871 = stablehlo.reshape %870 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %872 = stablehlo.multiply %150, %140 : tensor<256xf64>
      %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %873 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %874 = stablehlo.add %873, %872 : tensor<256xf64>
      %875 = stablehlo.multiply %225, %211 : tensor<256xf64>
      %876 = stablehlo.add %874, %875 : tensor<256xf64>
      %877 = stablehlo.multiply %313, %295 : tensor<256xf64>
      %878 = stablehlo.add %876, %877 : tensor<256xf64>
      %879 = stablehlo.multiply %410, %388 : tensor<256xf64>
      %880 = stablehlo.add %878, %879 : tensor<256xf64>
      %881 = stablehlo.multiply %512, %486 : tensor<256xf64>
      %882 = stablehlo.add %880, %881 : tensor<256xf64>
      %883 = stablehlo.multiply %615, %585 : tensor<256xf64>
      %884 = stablehlo.add %882, %883 : tensor<256xf64>
      %885 = stablehlo.multiply %715, %681 : tensor<256xf64>
      %886 = stablehlo.add %884, %885 : tensor<256xf64>
      %887 = stablehlo.multiply %808, %770 : tensor<256xf64>
      %888 = stablehlo.add %886, %887 : tensor<256xf64>
      %889 = stablehlo.subtract %871, %888 : tensor<256xf64>
      %890 = stablehlo.divide %889, %848 : tensor<256xf64>
      %891 = stablehlo.slice %92 [0:256, 11:12, 8:9] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %892 = stablehlo.reshape %891 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %893 = stablehlo.multiply %155, %140 : tensor<256xf64>
      %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %894 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %895 = stablehlo.add %894, %893 : tensor<256xf64>
      %896 = stablehlo.multiply %232, %211 : tensor<256xf64>
      %897 = stablehlo.add %895, %896 : tensor<256xf64>
      %898 = stablehlo.multiply %322, %295 : tensor<256xf64>
      %899 = stablehlo.add %897, %898 : tensor<256xf64>
      %900 = stablehlo.multiply %421, %388 : tensor<256xf64>
      %901 = stablehlo.add %899, %900 : tensor<256xf64>
      %902 = stablehlo.multiply %525, %486 : tensor<256xf64>
      %903 = stablehlo.add %901, %902 : tensor<256xf64>
      %904 = stablehlo.multiply %630, %585 : tensor<256xf64>
      %905 = stablehlo.add %903, %904 : tensor<256xf64>
      %906 = stablehlo.multiply %732, %681 : tensor<256xf64>
      %907 = stablehlo.add %905, %906 : tensor<256xf64>
      %908 = stablehlo.multiply %827, %770 : tensor<256xf64>
      %909 = stablehlo.add %907, %908 : tensor<256xf64>
      %910 = stablehlo.subtract %892, %909 : tensor<256xf64>
      %911 = stablehlo.divide %910, %848 : tensor<256xf64>
      %912 = stablehlo.slice %92 [0:256, 9:10, 9:10] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %913 = stablehlo.reshape %912 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %914 = stablehlo.multiply %145, %145 : tensor<256xf64>
      %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %915 = stablehlo.broadcast_in_dim %cst_137, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %916 = stablehlo.add %915, %914 : tensor<256xf64>
      %917 = stablehlo.multiply %218, %218 : tensor<256xf64>
      %918 = stablehlo.add %916, %917 : tensor<256xf64>
      %919 = stablehlo.multiply %304, %304 : tensor<256xf64>
      %920 = stablehlo.add %918, %919 : tensor<256xf64>
      %921 = stablehlo.multiply %399, %399 : tensor<256xf64>
      %922 = stablehlo.add %920, %921 : tensor<256xf64>
      %923 = stablehlo.multiply %499, %499 : tensor<256xf64>
      %924 = stablehlo.add %922, %923 : tensor<256xf64>
      %925 = stablehlo.multiply %600, %600 : tensor<256xf64>
      %926 = stablehlo.add %924, %925 : tensor<256xf64>
      %927 = stablehlo.multiply %698, %698 : tensor<256xf64>
      %928 = stablehlo.add %926, %927 : tensor<256xf64>
      %929 = stablehlo.multiply %789, %789 : tensor<256xf64>
      %930 = stablehlo.add %928, %929 : tensor<256xf64>
      %931 = stablehlo.multiply %869, %869 : tensor<256xf64>
      %932 = stablehlo.add %930, %931 : tensor<256xf64>
      %933 = stablehlo.subtract %913, %932 : tensor<256xf64>
      %934 = stablehlo.sqrt %933 : tensor<256xf64>
      %935 = stablehlo.slice %92 [0:256, 10:11, 9:10] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %936 = stablehlo.reshape %935 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %937 = stablehlo.multiply %150, %145 : tensor<256xf64>
      %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %938 = stablehlo.broadcast_in_dim %cst_138, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %939 = stablehlo.add %938, %937 : tensor<256xf64>
      %940 = stablehlo.multiply %225, %218 : tensor<256xf64>
      %941 = stablehlo.add %939, %940 : tensor<256xf64>
      %942 = stablehlo.multiply %313, %304 : tensor<256xf64>
      %943 = stablehlo.add %941, %942 : tensor<256xf64>
      %944 = stablehlo.multiply %410, %399 : tensor<256xf64>
      %945 = stablehlo.add %943, %944 : tensor<256xf64>
      %946 = stablehlo.multiply %512, %499 : tensor<256xf64>
      %947 = stablehlo.add %945, %946 : tensor<256xf64>
      %948 = stablehlo.multiply %615, %600 : tensor<256xf64>
      %949 = stablehlo.add %947, %948 : tensor<256xf64>
      %950 = stablehlo.multiply %715, %698 : tensor<256xf64>
      %951 = stablehlo.add %949, %950 : tensor<256xf64>
      %952 = stablehlo.multiply %808, %789 : tensor<256xf64>
      %953 = stablehlo.add %951, %952 : tensor<256xf64>
      %954 = stablehlo.multiply %890, %869 : tensor<256xf64>
      %955 = stablehlo.add %953, %954 : tensor<256xf64>
      %956 = stablehlo.subtract %936, %955 : tensor<256xf64>
      %957 = stablehlo.divide %956, %934 : tensor<256xf64>
      %958 = stablehlo.slice %92 [0:256, 11:12, 9:10] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %959 = stablehlo.reshape %958 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %960 = stablehlo.multiply %155, %145 : tensor<256xf64>
      %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %961 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %962 = stablehlo.add %961, %960 : tensor<256xf64>
      %963 = stablehlo.multiply %232, %218 : tensor<256xf64>
      %964 = stablehlo.add %962, %963 : tensor<256xf64>
      %965 = stablehlo.multiply %322, %304 : tensor<256xf64>
      %966 = stablehlo.add %964, %965 : tensor<256xf64>
      %967 = stablehlo.multiply %421, %399 : tensor<256xf64>
      %968 = stablehlo.add %966, %967 : tensor<256xf64>
      %969 = stablehlo.multiply %525, %499 : tensor<256xf64>
      %970 = stablehlo.add %968, %969 : tensor<256xf64>
      %971 = stablehlo.multiply %630, %600 : tensor<256xf64>
      %972 = stablehlo.add %970, %971 : tensor<256xf64>
      %973 = stablehlo.multiply %732, %698 : tensor<256xf64>
      %974 = stablehlo.add %972, %973 : tensor<256xf64>
      %975 = stablehlo.multiply %827, %789 : tensor<256xf64>
      %976 = stablehlo.add %974, %975 : tensor<256xf64>
      %977 = stablehlo.multiply %911, %869 : tensor<256xf64>
      %978 = stablehlo.add %976, %977 : tensor<256xf64>
      %979 = stablehlo.subtract %959, %978 : tensor<256xf64>
      %980 = stablehlo.divide %979, %934 : tensor<256xf64>
      %981 = stablehlo.slice %92 [0:256, 10:11, 10:11] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %982 = stablehlo.reshape %981 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %983 = stablehlo.multiply %150, %150 : tensor<256xf64>
      %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %984 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %985 = stablehlo.add %984, %983 : tensor<256xf64>
      %986 = stablehlo.multiply %225, %225 : tensor<256xf64>
      %987 = stablehlo.add %985, %986 : tensor<256xf64>
      %988 = stablehlo.multiply %313, %313 : tensor<256xf64>
      %989 = stablehlo.add %987, %988 : tensor<256xf64>
      %990 = stablehlo.multiply %410, %410 : tensor<256xf64>
      %991 = stablehlo.add %989, %990 : tensor<256xf64>
      %992 = stablehlo.multiply %512, %512 : tensor<256xf64>
      %993 = stablehlo.add %991, %992 : tensor<256xf64>
      %994 = stablehlo.multiply %615, %615 : tensor<256xf64>
      %995 = stablehlo.add %993, %994 : tensor<256xf64>
      %996 = stablehlo.multiply %715, %715 : tensor<256xf64>
      %997 = stablehlo.add %995, %996 : tensor<256xf64>
      %998 = stablehlo.multiply %808, %808 : tensor<256xf64>
      %999 = stablehlo.add %997, %998 : tensor<256xf64>
      %1000 = stablehlo.multiply %890, %890 : tensor<256xf64>
      %1001 = stablehlo.add %999, %1000 : tensor<256xf64>
      %1002 = stablehlo.multiply %957, %957 : tensor<256xf64>
      %1003 = stablehlo.add %1001, %1002 : tensor<256xf64>
      %1004 = stablehlo.subtract %982, %1003 : tensor<256xf64>
      %1005 = stablehlo.sqrt %1004 : tensor<256xf64>
      %1006 = stablehlo.slice %92 [0:256, 11:12, 10:11] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %1007 = stablehlo.reshape %1006 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %1008 = stablehlo.multiply %155, %150 : tensor<256xf64>
      %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1009 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1010 = stablehlo.add %1009, %1008 : tensor<256xf64>
      %1011 = stablehlo.multiply %232, %225 : tensor<256xf64>
      %1012 = stablehlo.add %1010, %1011 : tensor<256xf64>
      %1013 = stablehlo.multiply %322, %313 : tensor<256xf64>
      %1014 = stablehlo.add %1012, %1013 : tensor<256xf64>
      %1015 = stablehlo.multiply %421, %410 : tensor<256xf64>
      %1016 = stablehlo.add %1014, %1015 : tensor<256xf64>
      %1017 = stablehlo.multiply %525, %512 : tensor<256xf64>
      %1018 = stablehlo.add %1016, %1017 : tensor<256xf64>
      %1019 = stablehlo.multiply %630, %615 : tensor<256xf64>
      %1020 = stablehlo.add %1018, %1019 : tensor<256xf64>
      %1021 = stablehlo.multiply %732, %715 : tensor<256xf64>
      %1022 = stablehlo.add %1020, %1021 : tensor<256xf64>
      %1023 = stablehlo.multiply %827, %808 : tensor<256xf64>
      %1024 = stablehlo.add %1022, %1023 : tensor<256xf64>
      %1025 = stablehlo.multiply %911, %890 : tensor<256xf64>
      %1026 = stablehlo.add %1024, %1025 : tensor<256xf64>
      %1027 = stablehlo.multiply %980, %957 : tensor<256xf64>
      %1028 = stablehlo.add %1026, %1027 : tensor<256xf64>
      %1029 = stablehlo.subtract %1007, %1028 : tensor<256xf64>
      %1030 = stablehlo.divide %1029, %1005 : tensor<256xf64>
      %1031 = stablehlo.slice %92 [0:256, 11:12, 11:12] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
      %1032 = stablehlo.reshape %1031 : (tensor<256x1x1xf64>) -> tensor<256xf64>
      %1033 = stablehlo.multiply %155, %155 : tensor<256xf64>
      %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1034 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1035 = stablehlo.add %1034, %1033 : tensor<256xf64>
      %1036 = stablehlo.multiply %232, %232 : tensor<256xf64>
      %1037 = stablehlo.add %1035, %1036 : tensor<256xf64>
      %1038 = stablehlo.multiply %322, %322 : tensor<256xf64>
      %1039 = stablehlo.add %1037, %1038 : tensor<256xf64>
      %1040 = stablehlo.multiply %421, %421 : tensor<256xf64>
      %1041 = stablehlo.add %1039, %1040 : tensor<256xf64>
      %1042 = stablehlo.multiply %525, %525 : tensor<256xf64>
      %1043 = stablehlo.add %1041, %1042 : tensor<256xf64>
      %1044 = stablehlo.multiply %630, %630 : tensor<256xf64>
      %1045 = stablehlo.add %1043, %1044 : tensor<256xf64>
      %1046 = stablehlo.multiply %732, %732 : tensor<256xf64>
      %1047 = stablehlo.add %1045, %1046 : tensor<256xf64>
      %1048 = stablehlo.multiply %827, %827 : tensor<256xf64>
      %1049 = stablehlo.add %1047, %1048 : tensor<256xf64>
      %1050 = stablehlo.multiply %911, %911 : tensor<256xf64>
      %1051 = stablehlo.add %1049, %1050 : tensor<256xf64>
      %1052 = stablehlo.multiply %980, %980 : tensor<256xf64>
      %1053 = stablehlo.add %1051, %1052 : tensor<256xf64>
      %1054 = stablehlo.multiply %1030, %1030 : tensor<256xf64>
      %1055 = stablehlo.add %1053, %1054 : tensor<256xf64>
      %1056 = stablehlo.subtract %1032, %1055 : tensor<256xf64>
      %1057 = stablehlo.sqrt %1056 : tensor<256xf64>
      %1058 = stablehlo.slice %95 [0:1, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1059 = stablehlo.transpose %1058, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1060 = stablehlo.reshape %1059 : (tensor<256x1xf64>) -> tensor<256xf64>
      %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1061 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1062 = stablehlo.subtract %1060, %1061 : tensor<256xf64>
      %1063 = stablehlo.divide %1062, %100 : tensor<256xf64>
      %1064 = stablehlo.slice %95 [1:2, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1065 = stablehlo.transpose %1064, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1066 = stablehlo.reshape %1065 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1067 = stablehlo.multiply %105, %1063 : tensor<256xf64>
      %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1068 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1069 = stablehlo.add %1068, %1067 : tensor<256xf64>
      %1070 = stablehlo.subtract %1066, %1069 : tensor<256xf64>
      %1071 = stablehlo.divide %1070, %162 : tensor<256xf64>
      %1072 = stablehlo.slice %95 [2:3, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1073 = stablehlo.transpose %1072, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1074 = stablehlo.reshape %1073 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1075 = stablehlo.multiply %110, %1063 : tensor<256xf64>
      %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1076 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1077 = stablehlo.add %1076, %1075 : tensor<256xf64>
      %1078 = stablehlo.multiply %169, %1071 : tensor<256xf64>
      %1079 = stablehlo.add %1077, %1078 : tensor<256xf64>
      %1080 = stablehlo.subtract %1074, %1079 : tensor<256xf64>
      %1081 = stablehlo.divide %1080, %241 : tensor<256xf64>
      %1082 = stablehlo.slice %95 [3:4, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1083 = stablehlo.transpose %1082, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1084 = stablehlo.reshape %1083 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1085 = stablehlo.multiply %115, %1063 : tensor<256xf64>
      %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1086 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1087 = stablehlo.add %1086, %1085 : tensor<256xf64>
      %1088 = stablehlo.multiply %176, %1071 : tensor<256xf64>
      %1089 = stablehlo.add %1087, %1088 : tensor<256xf64>
      %1090 = stablehlo.multiply %250, %1081 : tensor<256xf64>
      %1091 = stablehlo.add %1089, %1090 : tensor<256xf64>
      %1092 = stablehlo.subtract %1084, %1091 : tensor<256xf64>
      %1093 = stablehlo.divide %1092, %333 : tensor<256xf64>
      %1094 = stablehlo.slice %95 [4:5, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1095 = stablehlo.transpose %1094, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1096 = stablehlo.reshape %1095 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1097 = stablehlo.multiply %120, %1063 : tensor<256xf64>
      %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1098 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1099 = stablehlo.add %1098, %1097 : tensor<256xf64>
      %1100 = stablehlo.multiply %183, %1071 : tensor<256xf64>
      %1101 = stablehlo.add %1099, %1100 : tensor<256xf64>
      %1102 = stablehlo.multiply %259, %1081 : tensor<256xf64>
      %1103 = stablehlo.add %1101, %1102 : tensor<256xf64>
      %1104 = stablehlo.multiply %344, %1093 : tensor<256xf64>
      %1105 = stablehlo.add %1103, %1104 : tensor<256xf64>
      %1106 = stablehlo.subtract %1096, %1105 : tensor<256xf64>
      %1107 = stablehlo.divide %1106, %434 : tensor<256xf64>
      %1108 = stablehlo.slice %95 [5:6, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1109 = stablehlo.transpose %1108, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1110 = stablehlo.reshape %1109 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1111 = stablehlo.multiply %125, %1063 : tensor<256xf64>
      %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1112 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1113 = stablehlo.add %1112, %1111 : tensor<256xf64>
      %1114 = stablehlo.multiply %190, %1071 : tensor<256xf64>
      %1115 = stablehlo.add %1113, %1114 : tensor<256xf64>
      %1116 = stablehlo.multiply %268, %1081 : tensor<256xf64>
      %1117 = stablehlo.add %1115, %1116 : tensor<256xf64>
      %1118 = stablehlo.multiply %355, %1093 : tensor<256xf64>
      %1119 = stablehlo.add %1117, %1118 : tensor<256xf64>
      %1120 = stablehlo.multiply %447, %1107 : tensor<256xf64>
      %1121 = stablehlo.add %1119, %1120 : tensor<256xf64>
      %1122 = stablehlo.subtract %1110, %1121 : tensor<256xf64>
      %1123 = stablehlo.divide %1122, %540 : tensor<256xf64>
      %1124 = stablehlo.slice %95 [6:7, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1125 = stablehlo.transpose %1124, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1126 = stablehlo.reshape %1125 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1127 = stablehlo.multiply %130, %1063 : tensor<256xf64>
      %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1128 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1129 = stablehlo.add %1128, %1127 : tensor<256xf64>
      %1130 = stablehlo.multiply %197, %1071 : tensor<256xf64>
      %1131 = stablehlo.add %1129, %1130 : tensor<256xf64>
      %1132 = stablehlo.multiply %277, %1081 : tensor<256xf64>
      %1133 = stablehlo.add %1131, %1132 : tensor<256xf64>
      %1134 = stablehlo.multiply %366, %1093 : tensor<256xf64>
      %1135 = stablehlo.add %1133, %1134 : tensor<256xf64>
      %1136 = stablehlo.multiply %460, %1107 : tensor<256xf64>
      %1137 = stablehlo.add %1135, %1136 : tensor<256xf64>
      %1138 = stablehlo.multiply %555, %1123 : tensor<256xf64>
      %1139 = stablehlo.add %1137, %1138 : tensor<256xf64>
      %1140 = stablehlo.subtract %1126, %1139 : tensor<256xf64>
      %1141 = stablehlo.divide %1140, %647 : tensor<256xf64>
      %1142 = stablehlo.slice %95 [7:8, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1143 = stablehlo.transpose %1142, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1144 = stablehlo.reshape %1143 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1145 = stablehlo.multiply %135, %1063 : tensor<256xf64>
      %cst_150 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1146 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1147 = stablehlo.add %1146, %1145 : tensor<256xf64>
      %1148 = stablehlo.multiply %204, %1071 : tensor<256xf64>
      %1149 = stablehlo.add %1147, %1148 : tensor<256xf64>
      %1150 = stablehlo.multiply %286, %1081 : tensor<256xf64>
      %1151 = stablehlo.add %1149, %1150 : tensor<256xf64>
      %1152 = stablehlo.multiply %377, %1093 : tensor<256xf64>
      %1153 = stablehlo.add %1151, %1152 : tensor<256xf64>
      %1154 = stablehlo.multiply %473, %1107 : tensor<256xf64>
      %1155 = stablehlo.add %1153, %1154 : tensor<256xf64>
      %1156 = stablehlo.multiply %570, %1123 : tensor<256xf64>
      %1157 = stablehlo.add %1155, %1156 : tensor<256xf64>
      %1158 = stablehlo.multiply %664, %1141 : tensor<256xf64>
      %1159 = stablehlo.add %1157, %1158 : tensor<256xf64>
      %1160 = stablehlo.subtract %1144, %1159 : tensor<256xf64>
      %1161 = stablehlo.divide %1160, %751 : tensor<256xf64>
      %1162 = stablehlo.slice %95 [8:9, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1163 = stablehlo.transpose %1162, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1164 = stablehlo.reshape %1163 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1165 = stablehlo.multiply %140, %1063 : tensor<256xf64>
      %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1166 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1167 = stablehlo.add %1166, %1165 : tensor<256xf64>
      %1168 = stablehlo.multiply %211, %1071 : tensor<256xf64>
      %1169 = stablehlo.add %1167, %1168 : tensor<256xf64>
      %1170 = stablehlo.multiply %295, %1081 : tensor<256xf64>
      %1171 = stablehlo.add %1169, %1170 : tensor<256xf64>
      %1172 = stablehlo.multiply %388, %1093 : tensor<256xf64>
      %1173 = stablehlo.add %1171, %1172 : tensor<256xf64>
      %1174 = stablehlo.multiply %486, %1107 : tensor<256xf64>
      %1175 = stablehlo.add %1173, %1174 : tensor<256xf64>
      %1176 = stablehlo.multiply %585, %1123 : tensor<256xf64>
      %1177 = stablehlo.add %1175, %1176 : tensor<256xf64>
      %1178 = stablehlo.multiply %681, %1141 : tensor<256xf64>
      %1179 = stablehlo.add %1177, %1178 : tensor<256xf64>
      %1180 = stablehlo.multiply %770, %1161 : tensor<256xf64>
      %1181 = stablehlo.add %1179, %1180 : tensor<256xf64>
      %1182 = stablehlo.subtract %1164, %1181 : tensor<256xf64>
      %1183 = stablehlo.divide %1182, %848 : tensor<256xf64>
      %1184 = stablehlo.slice %95 [9:10, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1185 = stablehlo.transpose %1184, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1186 = stablehlo.reshape %1185 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1187 = stablehlo.multiply %145, %1063 : tensor<256xf64>
      %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1188 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1189 = stablehlo.add %1188, %1187 : tensor<256xf64>
      %1190 = stablehlo.multiply %218, %1071 : tensor<256xf64>
      %1191 = stablehlo.add %1189, %1190 : tensor<256xf64>
      %1192 = stablehlo.multiply %304, %1081 : tensor<256xf64>
      %1193 = stablehlo.add %1191, %1192 : tensor<256xf64>
      %1194 = stablehlo.multiply %399, %1093 : tensor<256xf64>
      %1195 = stablehlo.add %1193, %1194 : tensor<256xf64>
      %1196 = stablehlo.multiply %499, %1107 : tensor<256xf64>
      %1197 = stablehlo.add %1195, %1196 : tensor<256xf64>
      %1198 = stablehlo.multiply %600, %1123 : tensor<256xf64>
      %1199 = stablehlo.add %1197, %1198 : tensor<256xf64>
      %1200 = stablehlo.multiply %698, %1141 : tensor<256xf64>
      %1201 = stablehlo.add %1199, %1200 : tensor<256xf64>
      %1202 = stablehlo.multiply %789, %1161 : tensor<256xf64>
      %1203 = stablehlo.add %1201, %1202 : tensor<256xf64>
      %1204 = stablehlo.multiply %869, %1183 : tensor<256xf64>
      %1205 = stablehlo.add %1203, %1204 : tensor<256xf64>
      %1206 = stablehlo.subtract %1186, %1205 : tensor<256xf64>
      %1207 = stablehlo.divide %1206, %934 : tensor<256xf64>
      %1208 = stablehlo.slice %95 [10:11, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1209 = stablehlo.transpose %1208, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1210 = stablehlo.reshape %1209 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1211 = stablehlo.multiply %150, %1063 : tensor<256xf64>
      %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1212 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1213 = stablehlo.add %1212, %1211 : tensor<256xf64>
      %1214 = stablehlo.multiply %225, %1071 : tensor<256xf64>
      %1215 = stablehlo.add %1213, %1214 : tensor<256xf64>
      %1216 = stablehlo.multiply %313, %1081 : tensor<256xf64>
      %1217 = stablehlo.add %1215, %1216 : tensor<256xf64>
      %1218 = stablehlo.multiply %410, %1093 : tensor<256xf64>
      %1219 = stablehlo.add %1217, %1218 : tensor<256xf64>
      %1220 = stablehlo.multiply %512, %1107 : tensor<256xf64>
      %1221 = stablehlo.add %1219, %1220 : tensor<256xf64>
      %1222 = stablehlo.multiply %615, %1123 : tensor<256xf64>
      %1223 = stablehlo.add %1221, %1222 : tensor<256xf64>
      %1224 = stablehlo.multiply %715, %1141 : tensor<256xf64>
      %1225 = stablehlo.add %1223, %1224 : tensor<256xf64>
      %1226 = stablehlo.multiply %808, %1161 : tensor<256xf64>
      %1227 = stablehlo.add %1225, %1226 : tensor<256xf64>
      %1228 = stablehlo.multiply %890, %1183 : tensor<256xf64>
      %1229 = stablehlo.add %1227, %1228 : tensor<256xf64>
      %1230 = stablehlo.multiply %957, %1207 : tensor<256xf64>
      %1231 = stablehlo.add %1229, %1230 : tensor<256xf64>
      %1232 = stablehlo.subtract %1210, %1231 : tensor<256xf64>
      %1233 = stablehlo.divide %1232, %1005 : tensor<256xf64>
      %1234 = stablehlo.slice %95 [11:12, 0:256] : (tensor<12x256xf64>) -> tensor<1x256xf64>
      %1235 = stablehlo.transpose %1234, dims = [1, 0] : (tensor<1x256xf64>) -> tensor<256x1xf64>
      %1236 = stablehlo.reshape %1235 : (tensor<256x1xf64>) -> tensor<256xf64>
      %1237 = stablehlo.multiply %155, %1063 : tensor<256xf64>
      %cst_154 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1238 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1239 = stablehlo.add %1238, %1237 : tensor<256xf64>
      %1240 = stablehlo.multiply %232, %1071 : tensor<256xf64>
      %1241 = stablehlo.add %1239, %1240 : tensor<256xf64>
      %1242 = stablehlo.multiply %322, %1081 : tensor<256xf64>
      %1243 = stablehlo.add %1241, %1242 : tensor<256xf64>
      %1244 = stablehlo.multiply %421, %1093 : tensor<256xf64>
      %1245 = stablehlo.add %1243, %1244 : tensor<256xf64>
      %1246 = stablehlo.multiply %525, %1107 : tensor<256xf64>
      %1247 = stablehlo.add %1245, %1246 : tensor<256xf64>
      %1248 = stablehlo.multiply %630, %1123 : tensor<256xf64>
      %1249 = stablehlo.add %1247, %1248 : tensor<256xf64>
      %1250 = stablehlo.multiply %732, %1141 : tensor<256xf64>
      %1251 = stablehlo.add %1249, %1250 : tensor<256xf64>
      %1252 = stablehlo.multiply %827, %1161 : tensor<256xf64>
      %1253 = stablehlo.add %1251, %1252 : tensor<256xf64>
      %1254 = stablehlo.multiply %911, %1183 : tensor<256xf64>
      %1255 = stablehlo.add %1253, %1254 : tensor<256xf64>
      %1256 = stablehlo.multiply %980, %1207 : tensor<256xf64>
      %1257 = stablehlo.add %1255, %1256 : tensor<256xf64>
      %1258 = stablehlo.multiply %1030, %1233 : tensor<256xf64>
      %1259 = stablehlo.add %1257, %1258 : tensor<256xf64>
      %1260 = stablehlo.subtract %1236, %1259 : tensor<256xf64>
      %1261 = stablehlo.divide %1260, %1057 : tensor<256xf64>
      %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1262 = stablehlo.broadcast_in_dim %cst_155, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1263 = stablehlo.subtract %1261, %1262 : tensor<256xf64>
      %1264 = stablehlo.divide %1263, %1057 : tensor<256xf64>
      %1265 = stablehlo.multiply %1030, %1264 : tensor<256xf64>
      %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1266 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1267 = stablehlo.add %1266, %1265 : tensor<256xf64>
      %1268 = stablehlo.subtract %1233, %1267 : tensor<256xf64>
      %1269 = stablehlo.divide %1268, %1005 : tensor<256xf64>
      %1270 = stablehlo.multiply %957, %1269 : tensor<256xf64>
      %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1271 = stablehlo.broadcast_in_dim %cst_157, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1272 = stablehlo.add %1271, %1270 : tensor<256xf64>
      %1273 = stablehlo.multiply %980, %1264 : tensor<256xf64>
      %1274 = stablehlo.add %1272, %1273 : tensor<256xf64>
      %1275 = stablehlo.subtract %1207, %1274 : tensor<256xf64>
      %1276 = stablehlo.divide %1275, %934 : tensor<256xf64>
      %1277 = stablehlo.multiply %869, %1276 : tensor<256xf64>
      %cst_158 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1278 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1279 = stablehlo.add %1278, %1277 : tensor<256xf64>
      %1280 = stablehlo.multiply %890, %1269 : tensor<256xf64>
      %1281 = stablehlo.add %1279, %1280 : tensor<256xf64>
      %1282 = stablehlo.multiply %911, %1264 : tensor<256xf64>
      %1283 = stablehlo.add %1281, %1282 : tensor<256xf64>
      %1284 = stablehlo.subtract %1183, %1283 : tensor<256xf64>
      %1285 = stablehlo.divide %1284, %848 : tensor<256xf64>
      %1286 = stablehlo.multiply %770, %1285 : tensor<256xf64>
      %cst_159 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1287 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1288 = stablehlo.add %1287, %1286 : tensor<256xf64>
      %1289 = stablehlo.multiply %789, %1276 : tensor<256xf64>
      %1290 = stablehlo.add %1288, %1289 : tensor<256xf64>
      %1291 = stablehlo.multiply %808, %1269 : tensor<256xf64>
      %1292 = stablehlo.add %1290, %1291 : tensor<256xf64>
      %1293 = stablehlo.multiply %827, %1264 : tensor<256xf64>
      %1294 = stablehlo.add %1292, %1293 : tensor<256xf64>
      %1295 = stablehlo.subtract %1161, %1294 : tensor<256xf64>
      %1296 = stablehlo.divide %1295, %751 : tensor<256xf64>
      %1297 = stablehlo.multiply %664, %1296 : tensor<256xf64>
      %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1298 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1299 = stablehlo.add %1298, %1297 : tensor<256xf64>
      %1300 = stablehlo.multiply %681, %1285 : tensor<256xf64>
      %1301 = stablehlo.add %1299, %1300 : tensor<256xf64>
      %1302 = stablehlo.multiply %698, %1276 : tensor<256xf64>
      %1303 = stablehlo.add %1301, %1302 : tensor<256xf64>
      %1304 = stablehlo.multiply %715, %1269 : tensor<256xf64>
      %1305 = stablehlo.add %1303, %1304 : tensor<256xf64>
      %1306 = stablehlo.multiply %732, %1264 : tensor<256xf64>
      %1307 = stablehlo.add %1305, %1306 : tensor<256xf64>
      %1308 = stablehlo.subtract %1141, %1307 : tensor<256xf64>
      %1309 = stablehlo.divide %1308, %647 : tensor<256xf64>
      %1310 = stablehlo.multiply %555, %1309 : tensor<256xf64>
      %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1311 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1312 = stablehlo.add %1311, %1310 : tensor<256xf64>
      %1313 = stablehlo.multiply %570, %1296 : tensor<256xf64>
      %1314 = stablehlo.add %1312, %1313 : tensor<256xf64>
      %1315 = stablehlo.multiply %585, %1285 : tensor<256xf64>
      %1316 = stablehlo.add %1314, %1315 : tensor<256xf64>
      %1317 = stablehlo.multiply %600, %1276 : tensor<256xf64>
      %1318 = stablehlo.add %1316, %1317 : tensor<256xf64>
      %1319 = stablehlo.multiply %615, %1269 : tensor<256xf64>
      %1320 = stablehlo.add %1318, %1319 : tensor<256xf64>
      %1321 = stablehlo.multiply %630, %1264 : tensor<256xf64>
      %1322 = stablehlo.add %1320, %1321 : tensor<256xf64>
      %1323 = stablehlo.subtract %1123, %1322 : tensor<256xf64>
      %1324 = stablehlo.divide %1323, %540 : tensor<256xf64>
      %1325 = stablehlo.multiply %447, %1324 : tensor<256xf64>
      %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1326 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1327 = stablehlo.add %1326, %1325 : tensor<256xf64>
      %1328 = stablehlo.multiply %460, %1309 : tensor<256xf64>
      %1329 = stablehlo.add %1327, %1328 : tensor<256xf64>
      %1330 = stablehlo.multiply %473, %1296 : tensor<256xf64>
      %1331 = stablehlo.add %1329, %1330 : tensor<256xf64>
      %1332 = stablehlo.multiply %486, %1285 : tensor<256xf64>
      %1333 = stablehlo.add %1331, %1332 : tensor<256xf64>
      %1334 = stablehlo.multiply %499, %1276 : tensor<256xf64>
      %1335 = stablehlo.add %1333, %1334 : tensor<256xf64>
      %1336 = stablehlo.multiply %512, %1269 : tensor<256xf64>
      %1337 = stablehlo.add %1335, %1336 : tensor<256xf64>
      %1338 = stablehlo.multiply %525, %1264 : tensor<256xf64>
      %1339 = stablehlo.add %1337, %1338 : tensor<256xf64>
      %1340 = stablehlo.subtract %1107, %1339 : tensor<256xf64>
      %1341 = stablehlo.divide %1340, %434 : tensor<256xf64>
      %1342 = stablehlo.multiply %344, %1341 : tensor<256xf64>
      %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1343 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1344 = stablehlo.add %1343, %1342 : tensor<256xf64>
      %1345 = stablehlo.multiply %355, %1324 : tensor<256xf64>
      %1346 = stablehlo.add %1344, %1345 : tensor<256xf64>
      %1347 = stablehlo.multiply %366, %1309 : tensor<256xf64>
      %1348 = stablehlo.add %1346, %1347 : tensor<256xf64>
      %1349 = stablehlo.multiply %377, %1296 : tensor<256xf64>
      %1350 = stablehlo.add %1348, %1349 : tensor<256xf64>
      %1351 = stablehlo.multiply %388, %1285 : tensor<256xf64>
      %1352 = stablehlo.add %1350, %1351 : tensor<256xf64>
      %1353 = stablehlo.multiply %399, %1276 : tensor<256xf64>
      %1354 = stablehlo.add %1352, %1353 : tensor<256xf64>
      %1355 = stablehlo.multiply %410, %1269 : tensor<256xf64>
      %1356 = stablehlo.add %1354, %1355 : tensor<256xf64>
      %1357 = stablehlo.multiply %421, %1264 : tensor<256xf64>
      %1358 = stablehlo.add %1356, %1357 : tensor<256xf64>
      %1359 = stablehlo.subtract %1093, %1358 : tensor<256xf64>
      %1360 = stablehlo.divide %1359, %333 : tensor<256xf64>
      %1361 = stablehlo.multiply %250, %1360 : tensor<256xf64>
      %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1362 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1363 = stablehlo.add %1362, %1361 : tensor<256xf64>
      %1364 = stablehlo.multiply %259, %1341 : tensor<256xf64>
      %1365 = stablehlo.add %1363, %1364 : tensor<256xf64>
      %1366 = stablehlo.multiply %268, %1324 : tensor<256xf64>
      %1367 = stablehlo.add %1365, %1366 : tensor<256xf64>
      %1368 = stablehlo.multiply %277, %1309 : tensor<256xf64>
      %1369 = stablehlo.add %1367, %1368 : tensor<256xf64>
      %1370 = stablehlo.multiply %286, %1296 : tensor<256xf64>
      %1371 = stablehlo.add %1369, %1370 : tensor<256xf64>
      %1372 = stablehlo.multiply %295, %1285 : tensor<256xf64>
      %1373 = stablehlo.add %1371, %1372 : tensor<256xf64>
      %1374 = stablehlo.multiply %304, %1276 : tensor<256xf64>
      %1375 = stablehlo.add %1373, %1374 : tensor<256xf64>
      %1376 = stablehlo.multiply %313, %1269 : tensor<256xf64>
      %1377 = stablehlo.add %1375, %1376 : tensor<256xf64>
      %1378 = stablehlo.multiply %322, %1264 : tensor<256xf64>
      %1379 = stablehlo.add %1377, %1378 : tensor<256xf64>
      %1380 = stablehlo.subtract %1081, %1379 : tensor<256xf64>
      %1381 = stablehlo.divide %1380, %241 : tensor<256xf64>
      %1382 = stablehlo.multiply %169, %1381 : tensor<256xf64>
      %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1383 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1384 = stablehlo.add %1383, %1382 : tensor<256xf64>
      %1385 = stablehlo.multiply %176, %1360 : tensor<256xf64>
      %1386 = stablehlo.add %1384, %1385 : tensor<256xf64>
      %1387 = stablehlo.multiply %183, %1341 : tensor<256xf64>
      %1388 = stablehlo.add %1386, %1387 : tensor<256xf64>
      %1389 = stablehlo.multiply %190, %1324 : tensor<256xf64>
      %1390 = stablehlo.add %1388, %1389 : tensor<256xf64>
      %1391 = stablehlo.multiply %197, %1309 : tensor<256xf64>
      %1392 = stablehlo.add %1390, %1391 : tensor<256xf64>
      %1393 = stablehlo.multiply %204, %1296 : tensor<256xf64>
      %1394 = stablehlo.add %1392, %1393 : tensor<256xf64>
      %1395 = stablehlo.multiply %211, %1285 : tensor<256xf64>
      %1396 = stablehlo.add %1394, %1395 : tensor<256xf64>
      %1397 = stablehlo.multiply %218, %1276 : tensor<256xf64>
      %1398 = stablehlo.add %1396, %1397 : tensor<256xf64>
      %1399 = stablehlo.multiply %225, %1269 : tensor<256xf64>
      %1400 = stablehlo.add %1398, %1399 : tensor<256xf64>
      %1401 = stablehlo.multiply %232, %1264 : tensor<256xf64>
      %1402 = stablehlo.add %1400, %1401 : tensor<256xf64>
      %1403 = stablehlo.subtract %1071, %1402 : tensor<256xf64>
      %1404 = stablehlo.divide %1403, %162 : tensor<256xf64>
      %1405 = stablehlo.multiply %105, %1404 : tensor<256xf64>
      %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1406 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1407 = stablehlo.add %1406, %1405 : tensor<256xf64>
      %1408 = stablehlo.multiply %110, %1381 : tensor<256xf64>
      %1409 = stablehlo.add %1407, %1408 : tensor<256xf64>
      %1410 = stablehlo.multiply %115, %1360 : tensor<256xf64>
      %1411 = stablehlo.add %1409, %1410 : tensor<256xf64>
      %1412 = stablehlo.multiply %120, %1341 : tensor<256xf64>
      %1413 = stablehlo.add %1411, %1412 : tensor<256xf64>
      %1414 = stablehlo.multiply %125, %1324 : tensor<256xf64>
      %1415 = stablehlo.add %1413, %1414 : tensor<256xf64>
      %1416 = stablehlo.multiply %130, %1309 : tensor<256xf64>
      %1417 = stablehlo.add %1415, %1416 : tensor<256xf64>
      %1418 = stablehlo.multiply %135, %1296 : tensor<256xf64>
      %1419 = stablehlo.add %1417, %1418 : tensor<256xf64>
      %1420 = stablehlo.multiply %140, %1285 : tensor<256xf64>
      %1421 = stablehlo.add %1419, %1420 : tensor<256xf64>
      %1422 = stablehlo.multiply %145, %1276 : tensor<256xf64>
      %1423 = stablehlo.add %1421, %1422 : tensor<256xf64>
      %1424 = stablehlo.multiply %150, %1269 : tensor<256xf64>
      %1425 = stablehlo.add %1423, %1424 : tensor<256xf64>
      %1426 = stablehlo.multiply %155, %1264 : tensor<256xf64>
      %1427 = stablehlo.add %1425, %1426 : tensor<256xf64>
      %1428 = stablehlo.subtract %1063, %1427 : tensor<256xf64>
      %1429 = stablehlo.divide %1428, %100 : tensor<256xf64>
      %1430 = stablehlo.broadcast_in_dim %1429, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1431 = stablehlo.broadcast_in_dim %1404, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1432 = stablehlo.broadcast_in_dim %1381, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1433 = stablehlo.broadcast_in_dim %1360, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1434 = stablehlo.broadcast_in_dim %1341, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1435 = stablehlo.broadcast_in_dim %1324, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1436 = stablehlo.broadcast_in_dim %1309, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1437 = stablehlo.broadcast_in_dim %1296, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1438 = stablehlo.broadcast_in_dim %1285, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1439 = stablehlo.broadcast_in_dim %1276, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1440 = stablehlo.broadcast_in_dim %1269, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1441 = stablehlo.broadcast_in_dim %1264, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
      %1442 = stablehlo.concatenate %1430, %1431, %1432, %1433, %1434, %1435, %1436, %1437, %1438, %1439, %1440, %1441, dim = 0 : (tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>) -> tensor<12x256xf64>
      %c_167 = stablehlo.constant dense<false> : tensor<i1>
      %1443 = func.call @_where_2(%c_167, %18, %1442) : (tensor<i1>, tensor<256x12xf64>, tensor<12x256xf64>) -> tensor<256x12xf64>
      %cst_168 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1444 = stablehlo.broadcast_in_dim %cst_168, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1445 = stablehlo.compare GT, %17, %1444, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
      %cst_169 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1446 = func.call @_where_3(%1445, %17, %cst_169) : (tensor<256xi1>, tensor<256xf64>, tensor<f64>) -> tensor<256xf64>
      %cst_170 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1447 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1448 = stablehlo.divide %1447, %1446 : tensor<256xf64>
      %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1449 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
      %1450 = stablehlo.compare NE, %14, %1449, FLOAT : (tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xi1>
      %1451 = stablehlo.compare NE, %14, %14, FLOAT : (tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xi1>
      %1452 = stablehlo.or %1450, %1451 : tensor<256x192xi1>
      %1453 = stablehlo.log %14 : tensor<256x192xf64>
      %1454 = stablehlo.multiply %14, %1453 : tensor<256x192xf64>
      %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1455 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f64>) -> tensor<192xf64>
      %1456 = func.call @_where_4(%1452, %1454, %1455) : (tensor<256x192xi1>, tensor<256x192xf64>, tensor<192xf64>) -> tensor<256x192xf64>
      %1457 = stablehlo.broadcast_in_dim %1448, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
      %1458 = stablehlo.broadcast_in_dim %1457, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
      %1459 = stablehlo.add %14, %1458 : tensor<256x192xf64>
      %1460 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
      %1461 = stablehlo.broadcast_in_dim %1460, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
      %1462 = stablehlo.multiply %1461, %14 : tensor<256x192xf64>
      %1463 = stablehlo.log_plus_one %1462 : tensor<256x192xf64>
      %1464 = stablehlo.multiply %1459, %1463 : tensor<256x192xf64>
      %1465 = func.call @_where_5(%1445, %1464, %14) : (tensor<256xi1>, tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xf64>
      %1466 = stablehlo.subtract %1456, %1465 : tensor<256x192xf64>
      %1467 = stablehlo.multiply %16, %1466 : tensor<256x192xf64>
      %cst_173 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1468 = stablehlo.reduce(%1467 init: %cst_173) applies stablehlo.add across dimensions = [1] : (tensor<256x192xf64>, tensor<f64>) -> tensor<256xf64>
      %1469 = stablehlo.dot_general %iterArg_31, %1443, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<192x12xf64>, tensor<256x12xf64>) -> tensor<192x256xf64>
      %1470 = stablehlo.transpose %1469, dims = [1, 0] : (tensor<192x256xf64>) -> tensor<256x192xf64>
      %1471 = stablehlo.add %1470, %15 : tensor<256x192xf64>
      %1472 = stablehlo.exponential %1471 : tensor<256x192xf64>
      %cst_174 = stablehlo.constant dense<1.000000e-300> : tensor<f64>
      %1473 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
      %1474 = stablehlo.maximum %1472, %1473 : tensor<256x192xf64>
      %cst_175 = stablehlo.constant dense<-690.77552789821368> : tensor<f64>
      %cst_176 = stablehlo.constant dense<690.77552789821368> : tensor<f64>
      %1475 = func.call @clip(%1471, %cst_175, %cst_176) : (tensor<256x192xf64>, tensor<f64>, tensor<f64>) -> tensor<256x192xf64>
      %cst_177 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1476 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1477 = stablehlo.compare GT, %17, %1476, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
      %cst_178 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1478 = func.call @_where_3(%1477, %17, %cst_178) : (tensor<256xi1>, tensor<256xf64>, tensor<f64>) -> tensor<256xf64>
      %cst_179 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1479 = stablehlo.broadcast_in_dim %cst_179, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1480 = stablehlo.divide %1479, %1478 : tensor<256xf64>
      %1481 = stablehlo.broadcast_in_dim %1480, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
      %1482 = stablehlo.broadcast_in_dim %1481, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
      %1483 = stablehlo.add %14, %1482 : tensor<256x192xf64>
      %1484 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
      %1485 = stablehlo.broadcast_in_dim %1484, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
      %1486 = stablehlo.multiply %1485, %1474 : tensor<256x192xf64>
      %1487 = stablehlo.log_plus_one %1486 : tensor<256x192xf64>
      %1488 = stablehlo.multiply %1483, %1487 : tensor<256x192xf64>
      %1489 = func.call @_where_5(%1477, %1488, %1474) : (tensor<256xi1>, tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xf64>
      %1490 = stablehlo.multiply %14, %1475 : tensor<256x192xf64>
      %1491 = stablehlo.subtract %1489, %1490 : tensor<256x192xf64>
      %1492 = stablehlo.multiply %16, %1491 : tensor<256x192xf64>
      %cst_180 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
      %1493 = stablehlo.reduce(%1492 init: %cst_180) applies stablehlo.add across dimensions = [1] : (tensor<256x192xf64>, tensor<f64>) -> tensor<256xf64>
      %1494 = stablehlo.add %1468, %1493 : tensor<256xf64>
      %cst_181 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
      %1495 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f64>) -> tensor<256xf64>
      %1496 = stablehlo.multiply %1495, %1494 : tensor<256xf64>
      %c_182 = stablehlo.constant dense<0> : tensor<i64>
      %1497 = stablehlo.broadcast_in_dim %c_182, dims = [] : (tensor<i64>) -> tensor<256xi64>
      %c_183 = stablehlo.constant dense<false> : tensor<i1>
      %1498 = stablehlo.broadcast_in_dim %c_183, dims = [] : (tensor<i1>) -> tensor<256xi1>
      %1499:18 = stablehlo.while(%iterArg_185 = %iterArg_31, %iterArg_186 = %15, %iterArg_187 = %17, %iterArg_188 = %14, %iterArg_189 = %16, %iterArg_190 = %iterArg_40, %iterArg_191 = %iterArg_41, %iterArg_192 = %iterArg_42, %iterArg_193 = %iterArg_43, %iterArg_194 = %iterArg_44, %iterArg_195 = %iterArg_45, %iterArg_196 = %iterArg_46, %iterArg_197 = %iterArg_47, %iterArg_198 = %1468, %iterArg_199 = %1443, %iterArg_200 = %1496, %iterArg_201 = %1497, %iterArg_202 = %1498) : tensor<192x12xf64>, tensor<256x192xf64>, tensor<256xf64>, tensor<256x192xf64>, tensor<256x192xf64>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<256xf64>, tensor<256x12xf64>, tensor<256xf64>, tensor<256xi64>, tensor<256xi1>
      cond {
        %c_203 = stablehlo.constant dense<50> : tensor<i64>
        %1505 = stablehlo.broadcast_in_dim %c_203, dims = [] : (tensor<i64>) -> tensor<256xi64>
        %1506 = stablehlo.compare LT, %iterArg_201, %1505, SIGNED : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
        %1507 = stablehlo.not %iterArg_202 : tensor<256xi1>
        %c_204 = stablehlo.constant dense<false> : tensor<i1>
        %1508 = stablehlo.broadcast_in_dim %c_204, dims = [] : (tensor<i1>) -> tensor<256xi1>
        %1509 = stablehlo.compare NE, %1506, %1508, UNSIGNED : (tensor<256xi1>, tensor<256xi1>) -> tensor<256xi1>
        %1510 = stablehlo.convert %1509 : tensor<256xi1>
        %1511 = stablehlo.and %1510, %1507 : tensor<256xi1>
        %c_205 = stablehlo.constant dense<false> : tensor<i1>
        %1512 = stablehlo.reduce(%1511 init: %c_205) applies stablehlo.or across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
        stablehlo.return %1512 : tensor<i1>
      } do {
        %1505 = stablehlo.dot_general %iterArg_185, %iterArg_199, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<192x12xf64>, tensor<256x12xf64>) -> tensor<192x256xf64>
        %1506 = stablehlo.transpose %1505, dims = [1, 0] : (tensor<192x256xf64>) -> tensor<256x192xf64>
        %1507 = stablehlo.add %1506, %iterArg_186 : tensor<256x192xf64>
        %cst_203 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %1508 = stablehlo.broadcast_in_dim %cst_203, dims = [] : (tensor<f64>) -> tensor<192xf64>
        %1509 = stablehlo.exponential %1507 : tensor<256x192xf64>
        %1510 = stablehlo.broadcast_in_dim %1508, dims = [1] : (tensor<192xf64>) -> tensor<1x192xf64>
        %1511 = stablehlo.broadcast_in_dim %1510, dims = [0, 1] : (tensor<1x192xf64>) -> tensor<256x192xf64>
        %1512 = stablehlo.multiply %1511, %1509 : tensor<256x192xf64>
        %cst_204 = stablehlo.constant dense<1.000000e-300> : tensor<f64>
        %1513 = stablehlo.broadcast_in_dim %cst_204, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
        %1514 = stablehlo.maximum %1509, %1513 : tensor<256x192xf64>
        %cst_205 = stablehlo.constant dense<1.000000e-300> : tensor<f64>
        %1515 = stablehlo.broadcast_in_dim %cst_205, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
        %1516 = stablehlo.compare GT, %1509, %1515, FLOAT : (tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xi1>
        %cst_206 = stablehlo.constant dense<1.000000e-300> : tensor<f64>
        %1517 = stablehlo.broadcast_in_dim %cst_206, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
        %1518 = stablehlo.compare EQ, %1509, %1517, FLOAT : (tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xi1>
        %cst_207 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %1519 = stablehlo.broadcast_in_dim %cst_207, dims = [] : (tensor<f64>) -> tensor<192xf64>
        %cst_208 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1520 = stablehlo.broadcast_in_dim %cst_208, dims = [] : (tensor<f64>) -> tensor<192xf64>
        %cst_209 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
        %1521 = stablehlo.broadcast_in_dim %cst_209, dims = [] : (tensor<f64>) -> tensor<192xf64>
        %1522 = stablehlo.broadcast_in_dim %1520, dims = [1] : (tensor<192xf64>) -> tensor<256x192xf64>
        %1523 = stablehlo.broadcast_in_dim %1521, dims = [1] : (tensor<192xf64>) -> tensor<256x192xf64>
        %1524 = stablehlo.select %1518, %1523, %1522 : tensor<256x192xi1>, tensor<256x192xf64>
        %1525 = stablehlo.broadcast_in_dim %1519, dims = [1] : (tensor<192xf64>) -> tensor<256x192xf64>
        %1526 = stablehlo.select %1516, %1525, %1524 : tensor<256x192xi1>, tensor<256x192xf64>
        %1527 = stablehlo.multiply %1512, %1526 : tensor<256x192xf64>
        %1528 = stablehlo.exponential %1507 : tensor<256x192xf64>
        %1529 = stablehlo.broadcast_in_dim %1508, dims = [1] : (tensor<192xf64>) -> tensor<1x192xf64>
        %1530 = stablehlo.broadcast_in_dim %1529, dims = [0, 1] : (tensor<1x192xf64>) -> tensor<256x192xf64>
        %1531 = stablehlo.multiply %1530, %1528 : tensor<256x192xf64>
        %1532 = stablehlo.multiply %1514, %1514 : tensor<256x192xf64>
        %cst_210 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
        %1533 = stablehlo.broadcast_in_dim %cst_210, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
        %1534 = stablehlo.multiply %1533, %1514 : tensor<256x192xf64>
        %1535 = stablehlo.multiply %1527, %1534 : tensor<256x192xf64>
        %1536 = stablehlo.broadcast_in_dim %iterArg_187, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
        %1537 = stablehlo.broadcast_in_dim %1536, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
        %1538 = stablehlo.multiply %1537, %1532 : tensor<256x192xf64>
        %1539 = stablehlo.broadcast_in_dim %iterArg_187, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
        %1540 = stablehlo.broadcast_in_dim %1539, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
        %1541 = stablehlo.multiply %1540, %1535 : tensor<256x192xf64>
        %1542 = stablehlo.add %1514, %1538 : tensor<256x192xf64>
        %1543 = stablehlo.add %1527, %1541 : tensor<256x192xf64>
        %1544 = stablehlo.divide %1528, %1542 : tensor<256x192xf64>
        %1545 = stablehlo.divide %1531, %1542 : tensor<256x192xf64>
        %1546 = stablehlo.negate %1543 : tensor<256x192xf64>
        %1547 = stablehlo.multiply %1546, %1528 : tensor<256x192xf64>
        %1548 = stablehlo.multiply %1542, %1542 : tensor<256x192xf64>
        %cst_211 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %1549 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
        %1550 = stablehlo.divide %1549, %1548 : tensor<256x192xf64>
        %1551 = stablehlo.multiply %1547, %1550 : tensor<256x192xf64>
        %1552 = stablehlo.add %1545, %1551 : tensor<256x192xf64>
        %1553 = stablehlo.exponential %1507 : tensor<256x192xf64>
        %1554 = stablehlo.subtract %iterArg_188, %1514 : tensor<256x192xf64>
        %1555 = stablehlo.multiply %iterArg_189, %1554 : tensor<256x192xf64>
        %1556 = stablehlo.multiply %iterArg_189, %1553 : tensor<256x192xf64>
        %1557 = stablehlo.multiply %1556, %1544 : tensor<256x192xf64>
        %1558 = stablehlo.multiply %1555, %1552 : tensor<256x192xf64>
        %1559 = stablehlo.subtract %1557, %1558 : tensor<256x192xf64>
        %cst_212 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1560 = stablehlo.broadcast_in_dim %cst_212, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
        %1561 = stablehlo.compare GT, %1559, %1560, FLOAT : (tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xi1>
        %1562 = func.call @_where_6(%1561, %1559, %1557) : (tensor<256x192xi1>, tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xf64>
        %1563 = stablehlo.multiply %1555, %1544 : tensor<256x192xf64>
        %c_213 = stablehlo.constant dense<12> : tensor<i64>
        %1564 = stablehlo.broadcast_in_dim %c_213, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1565 = stablehlo.add %iterArg_190, %1564 : tensor<78xi64>
        %1566 = stablehlo.select %iterArg_191, %1565, %iterArg_190 : tensor<78xi1>, tensor<78xi64>
        %1567 = stablehlo.convert %1566 : (tensor<78xi64>) -> tensor<78xi32>
        %1568 = stablehlo.broadcast_in_dim %1567, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1569 = "stablehlo.gather"(%iterArg_185, %1568) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 192, 1>}> : (tensor<192x12xf64>, tensor<78x1xi32>) -> tensor<192x78xf64>
        %c_214 = stablehlo.constant dense<12> : tensor<i64>
        %1570 = stablehlo.broadcast_in_dim %c_214, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1571 = stablehlo.add %iterArg_192, %1570 : tensor<78xi64>
        %1572 = stablehlo.select %iterArg_193, %1571, %iterArg_192 : tensor<78xi1>, tensor<78xi64>
        %1573 = stablehlo.convert %1572 : (tensor<78xi64>) -> tensor<78xi32>
        %1574 = stablehlo.broadcast_in_dim %1573, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1575 = "stablehlo.gather"(%iterArg_185, %1574) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 192, 1>}> : (tensor<192x12xf64>, tensor<78x1xi32>) -> tensor<192x78xf64>
        %1576 = stablehlo.multiply %1569, %1575 : tensor<192x78xf64>
        %1577 = stablehlo.dot_general %1562, %1576, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x192xf64>, tensor<192x78xf64>) -> tensor<256x78xf64>
        %cst_215 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1578 = stablehlo.broadcast_in_dim %cst_215, dims = [] : (tensor<f64>) -> tensor<12x12xf64>
        %c_216 = stablehlo.constant dense<12> : tensor<i64>
        %1579 = stablehlo.broadcast_in_dim %c_216, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1580 = stablehlo.add %iterArg_190, %1579 : tensor<78xi64>
        %1581 = stablehlo.select %iterArg_194, %1580, %iterArg_190 : tensor<78xi1>, tensor<78xi64>
        %c_217 = stablehlo.constant dense<12> : tensor<i64>
        %1582 = stablehlo.broadcast_in_dim %c_217, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1583 = stablehlo.add %iterArg_192, %1582 : tensor<78xi64>
        %1584 = stablehlo.select %iterArg_195, %1583, %iterArg_192 : tensor<78xi1>, tensor<78xi64>
        %1585 = stablehlo.convert %1581 : (tensor<78xi64>) -> tensor<78xi32>
        %1586 = stablehlo.convert %1584 : (tensor<78xi64>) -> tensor<78xi32>
        %1587 = stablehlo.broadcast_in_dim %1585, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1588 = stablehlo.broadcast_in_dim %1586, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1589 = stablehlo.concatenate %1587, %1588, dim = 1 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x2xi32>
        %1590 = stablehlo.broadcast_in_dim %1578, dims = [1, 2] : (tensor<12x12xf64>) -> tensor<256x12x12xf64>
        %1591 = "stablehlo.scatter"(%1590, %1589, %1577) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
          stablehlo.return %arg7 : tensor<f64>
        }) : (tensor<256x12x12xf64>, tensor<78x2xi32>, tensor<256x78xf64>) -> tensor<256x12x12xf64>
        %c_218 = stablehlo.constant dense<12> : tensor<i64>
        %1592 = stablehlo.broadcast_in_dim %c_218, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1593 = stablehlo.add %iterArg_192, %1592 : tensor<78xi64>
        %1594 = stablehlo.select %iterArg_196, %1593, %iterArg_192 : tensor<78xi1>, tensor<78xi64>
        %c_219 = stablehlo.constant dense<12> : tensor<i64>
        %1595 = stablehlo.broadcast_in_dim %c_219, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1596 = stablehlo.add %iterArg_190, %1595 : tensor<78xi64>
        %1597 = stablehlo.select %iterArg_197, %1596, %iterArg_190 : tensor<78xi1>, tensor<78xi64>
        %1598 = stablehlo.convert %1594 : (tensor<78xi64>) -> tensor<78xi32>
        %1599 = stablehlo.convert %1597 : (tensor<78xi64>) -> tensor<78xi32>
        %1600 = stablehlo.broadcast_in_dim %1598, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1601 = stablehlo.broadcast_in_dim %1599, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1602 = stablehlo.concatenate %1600, %1601, dim = 1 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x2xi32>
        %1603 = "stablehlo.scatter"(%1591, %1602, %1577) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
          stablehlo.return %arg7 : tensor<f64>
        }) : (tensor<256x12x12xf64>, tensor<78x2xi32>, tensor<256x78xf64>) -> tensor<256x12x12xf64>
        %1604 = stablehlo.iota dim = 0 : tensor<12x12xi64>
        %1605 = stablehlo.iota dim = 1 : tensor<12x12xi64>
        %c_220 = stablehlo.constant dense<0> : tensor<i64>
        %1606 = stablehlo.broadcast_in_dim %c_220, dims = [] : (tensor<i64>) -> tensor<12x12xi64>
        %1607 = stablehlo.add %1604, %1606 : tensor<12x12xi64>
        %1608 = stablehlo.compare EQ, %1607, %1605, SIGNED : (tensor<12x12xi64>, tensor<12x12xi64>) -> tensor<12x12xi1>
        %1609 = stablehlo.convert %1608 : (tensor<12x12xi1>) -> tensor<12x12xf64>
        %cst_221 = stablehlo.constant dense<1.000000e-10> : tensor<f64>
        %1610 = stablehlo.broadcast_in_dim %cst_221, dims = [] : (tensor<f64>) -> tensor<12x12xf64>
        %1611 = stablehlo.multiply %1610, %1609 : tensor<12x12xf64>
        %1612 = stablehlo.broadcast_in_dim %1611, dims = [1, 2] : (tensor<12x12xf64>) -> tensor<1x12x12xf64>
        %1613 = stablehlo.broadcast_in_dim %1612, dims = [0, 1, 2] : (tensor<1x12x12xf64>) -> tensor<256x12x12xf64>
        %1614 = stablehlo.add %1603, %1613 : tensor<256x12x12xf64>
        %1615 = stablehlo.transpose %iterArg_185, dims = [1, 0] : (tensor<192x12xf64>) -> tensor<12x192xf64>
        %1616 = stablehlo.dot_general %1615, %1563, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<12x192xf64>, tensor<256x192xf64>) -> tensor<12x256xf64>
        %cst_222 = stablehlo.constant dense<1.000000e-10> : tensor<f64>
        %1617 = stablehlo.broadcast_in_dim %cst_222, dims = [] : (tensor<f64>) -> tensor<256x12xf64>
        %1618 = stablehlo.multiply %1617, %iterArg_199 : tensor<256x12xf64>
        %1619 = stablehlo.transpose %1616, dims = [1, 0] : (tensor<12x256xf64>) -> tensor<256x12xf64>
        %1620 = stablehlo.subtract %1619, %1618 : tensor<256x12xf64>
        %1621 = stablehlo.slice %1614 [0:256, 0:1, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1622 = stablehlo.reshape %1621 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_223 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1623 = stablehlo.broadcast_in_dim %cst_223, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1624 = stablehlo.subtract %1622, %1623 : tensor<256xf64>
        %1625 = stablehlo.sqrt %1624 : tensor<256xf64>
        %1626 = stablehlo.slice %1614 [0:256, 1:2, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1627 = stablehlo.reshape %1626 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_224 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1628 = stablehlo.broadcast_in_dim %cst_224, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1629 = stablehlo.subtract %1627, %1628 : tensor<256xf64>
        %1630 = stablehlo.divide %1629, %1625 : tensor<256xf64>
        %1631 = stablehlo.slice %1614 [0:256, 2:3, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1632 = stablehlo.reshape %1631 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_225 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1633 = stablehlo.broadcast_in_dim %cst_225, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1634 = stablehlo.subtract %1632, %1633 : tensor<256xf64>
        %1635 = stablehlo.divide %1634, %1625 : tensor<256xf64>
        %1636 = stablehlo.slice %1614 [0:256, 3:4, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1637 = stablehlo.reshape %1636 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_226 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1638 = stablehlo.broadcast_in_dim %cst_226, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1639 = stablehlo.subtract %1637, %1638 : tensor<256xf64>
        %1640 = stablehlo.divide %1639, %1625 : tensor<256xf64>
        %1641 = stablehlo.slice %1614 [0:256, 4:5, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1642 = stablehlo.reshape %1641 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_227 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1643 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1644 = stablehlo.subtract %1642, %1643 : tensor<256xf64>
        %1645 = stablehlo.divide %1644, %1625 : tensor<256xf64>
        %1646 = stablehlo.slice %1614 [0:256, 5:6, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1647 = stablehlo.reshape %1646 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_228 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1648 = stablehlo.broadcast_in_dim %cst_228, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1649 = stablehlo.subtract %1647, %1648 : tensor<256xf64>
        %1650 = stablehlo.divide %1649, %1625 : tensor<256xf64>
        %1651 = stablehlo.slice %1614 [0:256, 6:7, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1652 = stablehlo.reshape %1651 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_229 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1653 = stablehlo.broadcast_in_dim %cst_229, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1654 = stablehlo.subtract %1652, %1653 : tensor<256xf64>
        %1655 = stablehlo.divide %1654, %1625 : tensor<256xf64>
        %1656 = stablehlo.slice %1614 [0:256, 7:8, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1657 = stablehlo.reshape %1656 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_230 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1658 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1659 = stablehlo.subtract %1657, %1658 : tensor<256xf64>
        %1660 = stablehlo.divide %1659, %1625 : tensor<256xf64>
        %1661 = stablehlo.slice %1614 [0:256, 8:9, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1662 = stablehlo.reshape %1661 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_231 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1663 = stablehlo.broadcast_in_dim %cst_231, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1664 = stablehlo.subtract %1662, %1663 : tensor<256xf64>
        %1665 = stablehlo.divide %1664, %1625 : tensor<256xf64>
        %1666 = stablehlo.slice %1614 [0:256, 9:10, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1667 = stablehlo.reshape %1666 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_232 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1668 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1669 = stablehlo.subtract %1667, %1668 : tensor<256xf64>
        %1670 = stablehlo.divide %1669, %1625 : tensor<256xf64>
        %1671 = stablehlo.slice %1614 [0:256, 10:11, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1672 = stablehlo.reshape %1671 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_233 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1673 = stablehlo.broadcast_in_dim %cst_233, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1674 = stablehlo.subtract %1672, %1673 : tensor<256xf64>
        %1675 = stablehlo.divide %1674, %1625 : tensor<256xf64>
        %1676 = stablehlo.slice %1614 [0:256, 11:12, 0:1] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1677 = stablehlo.reshape %1676 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %cst_234 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1678 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1679 = stablehlo.subtract %1677, %1678 : tensor<256xf64>
        %1680 = stablehlo.divide %1679, %1625 : tensor<256xf64>
        %1681 = stablehlo.slice %1614 [0:256, 1:2, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1682 = stablehlo.reshape %1681 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1683 = stablehlo.multiply %1630, %1630 : tensor<256xf64>
        %cst_235 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1684 = stablehlo.broadcast_in_dim %cst_235, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1685 = stablehlo.add %1684, %1683 : tensor<256xf64>
        %1686 = stablehlo.subtract %1682, %1685 : tensor<256xf64>
        %1687 = stablehlo.sqrt %1686 : tensor<256xf64>
        %1688 = stablehlo.slice %1614 [0:256, 2:3, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1689 = stablehlo.reshape %1688 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1690 = stablehlo.multiply %1635, %1630 : tensor<256xf64>
        %cst_236 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1691 = stablehlo.broadcast_in_dim %cst_236, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1692 = stablehlo.add %1691, %1690 : tensor<256xf64>
        %1693 = stablehlo.subtract %1689, %1692 : tensor<256xf64>
        %1694 = stablehlo.divide %1693, %1687 : tensor<256xf64>
        %1695 = stablehlo.slice %1614 [0:256, 3:4, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1696 = stablehlo.reshape %1695 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1697 = stablehlo.multiply %1640, %1630 : tensor<256xf64>
        %cst_237 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1698 = stablehlo.broadcast_in_dim %cst_237, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1699 = stablehlo.add %1698, %1697 : tensor<256xf64>
        %1700 = stablehlo.subtract %1696, %1699 : tensor<256xf64>
        %1701 = stablehlo.divide %1700, %1687 : tensor<256xf64>
        %1702 = stablehlo.slice %1614 [0:256, 4:5, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1703 = stablehlo.reshape %1702 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1704 = stablehlo.multiply %1645, %1630 : tensor<256xf64>
        %cst_238 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1705 = stablehlo.broadcast_in_dim %cst_238, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1706 = stablehlo.add %1705, %1704 : tensor<256xf64>
        %1707 = stablehlo.subtract %1703, %1706 : tensor<256xf64>
        %1708 = stablehlo.divide %1707, %1687 : tensor<256xf64>
        %1709 = stablehlo.slice %1614 [0:256, 5:6, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1710 = stablehlo.reshape %1709 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1711 = stablehlo.multiply %1650, %1630 : tensor<256xf64>
        %cst_239 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1712 = stablehlo.broadcast_in_dim %cst_239, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1713 = stablehlo.add %1712, %1711 : tensor<256xf64>
        %1714 = stablehlo.subtract %1710, %1713 : tensor<256xf64>
        %1715 = stablehlo.divide %1714, %1687 : tensor<256xf64>
        %1716 = stablehlo.slice %1614 [0:256, 6:7, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1717 = stablehlo.reshape %1716 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1718 = stablehlo.multiply %1655, %1630 : tensor<256xf64>
        %cst_240 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1719 = stablehlo.broadcast_in_dim %cst_240, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1720 = stablehlo.add %1719, %1718 : tensor<256xf64>
        %1721 = stablehlo.subtract %1717, %1720 : tensor<256xf64>
        %1722 = stablehlo.divide %1721, %1687 : tensor<256xf64>
        %1723 = stablehlo.slice %1614 [0:256, 7:8, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1724 = stablehlo.reshape %1723 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1725 = stablehlo.multiply %1660, %1630 : tensor<256xf64>
        %cst_241 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1726 = stablehlo.broadcast_in_dim %cst_241, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1727 = stablehlo.add %1726, %1725 : tensor<256xf64>
        %1728 = stablehlo.subtract %1724, %1727 : tensor<256xf64>
        %1729 = stablehlo.divide %1728, %1687 : tensor<256xf64>
        %1730 = stablehlo.slice %1614 [0:256, 8:9, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1731 = stablehlo.reshape %1730 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1732 = stablehlo.multiply %1665, %1630 : tensor<256xf64>
        %cst_242 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1733 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1734 = stablehlo.add %1733, %1732 : tensor<256xf64>
        %1735 = stablehlo.subtract %1731, %1734 : tensor<256xf64>
        %1736 = stablehlo.divide %1735, %1687 : tensor<256xf64>
        %1737 = stablehlo.slice %1614 [0:256, 9:10, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1738 = stablehlo.reshape %1737 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1739 = stablehlo.multiply %1670, %1630 : tensor<256xf64>
        %cst_243 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1740 = stablehlo.broadcast_in_dim %cst_243, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1741 = stablehlo.add %1740, %1739 : tensor<256xf64>
        %1742 = stablehlo.subtract %1738, %1741 : tensor<256xf64>
        %1743 = stablehlo.divide %1742, %1687 : tensor<256xf64>
        %1744 = stablehlo.slice %1614 [0:256, 10:11, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1745 = stablehlo.reshape %1744 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1746 = stablehlo.multiply %1675, %1630 : tensor<256xf64>
        %cst_244 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1747 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1748 = stablehlo.add %1747, %1746 : tensor<256xf64>
        %1749 = stablehlo.subtract %1745, %1748 : tensor<256xf64>
        %1750 = stablehlo.divide %1749, %1687 : tensor<256xf64>
        %1751 = stablehlo.slice %1614 [0:256, 11:12, 1:2] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1752 = stablehlo.reshape %1751 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1753 = stablehlo.multiply %1680, %1630 : tensor<256xf64>
        %cst_245 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1754 = stablehlo.broadcast_in_dim %cst_245, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1755 = stablehlo.add %1754, %1753 : tensor<256xf64>
        %1756 = stablehlo.subtract %1752, %1755 : tensor<256xf64>
        %1757 = stablehlo.divide %1756, %1687 : tensor<256xf64>
        %1758 = stablehlo.slice %1614 [0:256, 2:3, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1759 = stablehlo.reshape %1758 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1760 = stablehlo.multiply %1635, %1635 : tensor<256xf64>
        %cst_246 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1761 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1762 = stablehlo.add %1761, %1760 : tensor<256xf64>
        %1763 = stablehlo.multiply %1694, %1694 : tensor<256xf64>
        %1764 = stablehlo.add %1762, %1763 : tensor<256xf64>
        %1765 = stablehlo.subtract %1759, %1764 : tensor<256xf64>
        %1766 = stablehlo.sqrt %1765 : tensor<256xf64>
        %1767 = stablehlo.slice %1614 [0:256, 3:4, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1768 = stablehlo.reshape %1767 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1769 = stablehlo.multiply %1640, %1635 : tensor<256xf64>
        %cst_247 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1770 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1771 = stablehlo.add %1770, %1769 : tensor<256xf64>
        %1772 = stablehlo.multiply %1701, %1694 : tensor<256xf64>
        %1773 = stablehlo.add %1771, %1772 : tensor<256xf64>
        %1774 = stablehlo.subtract %1768, %1773 : tensor<256xf64>
        %1775 = stablehlo.divide %1774, %1766 : tensor<256xf64>
        %1776 = stablehlo.slice %1614 [0:256, 4:5, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1777 = stablehlo.reshape %1776 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1778 = stablehlo.multiply %1645, %1635 : tensor<256xf64>
        %cst_248 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1779 = stablehlo.broadcast_in_dim %cst_248, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1780 = stablehlo.add %1779, %1778 : tensor<256xf64>
        %1781 = stablehlo.multiply %1708, %1694 : tensor<256xf64>
        %1782 = stablehlo.add %1780, %1781 : tensor<256xf64>
        %1783 = stablehlo.subtract %1777, %1782 : tensor<256xf64>
        %1784 = stablehlo.divide %1783, %1766 : tensor<256xf64>
        %1785 = stablehlo.slice %1614 [0:256, 5:6, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1786 = stablehlo.reshape %1785 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1787 = stablehlo.multiply %1650, %1635 : tensor<256xf64>
        %cst_249 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1788 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1789 = stablehlo.add %1788, %1787 : tensor<256xf64>
        %1790 = stablehlo.multiply %1715, %1694 : tensor<256xf64>
        %1791 = stablehlo.add %1789, %1790 : tensor<256xf64>
        %1792 = stablehlo.subtract %1786, %1791 : tensor<256xf64>
        %1793 = stablehlo.divide %1792, %1766 : tensor<256xf64>
        %1794 = stablehlo.slice %1614 [0:256, 6:7, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1795 = stablehlo.reshape %1794 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1796 = stablehlo.multiply %1655, %1635 : tensor<256xf64>
        %cst_250 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1797 = stablehlo.broadcast_in_dim %cst_250, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1798 = stablehlo.add %1797, %1796 : tensor<256xf64>
        %1799 = stablehlo.multiply %1722, %1694 : tensor<256xf64>
        %1800 = stablehlo.add %1798, %1799 : tensor<256xf64>
        %1801 = stablehlo.subtract %1795, %1800 : tensor<256xf64>
        %1802 = stablehlo.divide %1801, %1766 : tensor<256xf64>
        %1803 = stablehlo.slice %1614 [0:256, 7:8, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1804 = stablehlo.reshape %1803 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1805 = stablehlo.multiply %1660, %1635 : tensor<256xf64>
        %cst_251 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1806 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1807 = stablehlo.add %1806, %1805 : tensor<256xf64>
        %1808 = stablehlo.multiply %1729, %1694 : tensor<256xf64>
        %1809 = stablehlo.add %1807, %1808 : tensor<256xf64>
        %1810 = stablehlo.subtract %1804, %1809 : tensor<256xf64>
        %1811 = stablehlo.divide %1810, %1766 : tensor<256xf64>
        %1812 = stablehlo.slice %1614 [0:256, 8:9, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1813 = stablehlo.reshape %1812 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1814 = stablehlo.multiply %1665, %1635 : tensor<256xf64>
        %cst_252 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1815 = stablehlo.broadcast_in_dim %cst_252, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1816 = stablehlo.add %1815, %1814 : tensor<256xf64>
        %1817 = stablehlo.multiply %1736, %1694 : tensor<256xf64>
        %1818 = stablehlo.add %1816, %1817 : tensor<256xf64>
        %1819 = stablehlo.subtract %1813, %1818 : tensor<256xf64>
        %1820 = stablehlo.divide %1819, %1766 : tensor<256xf64>
        %1821 = stablehlo.slice %1614 [0:256, 9:10, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1822 = stablehlo.reshape %1821 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1823 = stablehlo.multiply %1670, %1635 : tensor<256xf64>
        %cst_253 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1824 = stablehlo.broadcast_in_dim %cst_253, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1825 = stablehlo.add %1824, %1823 : tensor<256xf64>
        %1826 = stablehlo.multiply %1743, %1694 : tensor<256xf64>
        %1827 = stablehlo.add %1825, %1826 : tensor<256xf64>
        %1828 = stablehlo.subtract %1822, %1827 : tensor<256xf64>
        %1829 = stablehlo.divide %1828, %1766 : tensor<256xf64>
        %1830 = stablehlo.slice %1614 [0:256, 10:11, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1831 = stablehlo.reshape %1830 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1832 = stablehlo.multiply %1675, %1635 : tensor<256xf64>
        %cst_254 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1833 = stablehlo.broadcast_in_dim %cst_254, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1834 = stablehlo.add %1833, %1832 : tensor<256xf64>
        %1835 = stablehlo.multiply %1750, %1694 : tensor<256xf64>
        %1836 = stablehlo.add %1834, %1835 : tensor<256xf64>
        %1837 = stablehlo.subtract %1831, %1836 : tensor<256xf64>
        %1838 = stablehlo.divide %1837, %1766 : tensor<256xf64>
        %1839 = stablehlo.slice %1614 [0:256, 11:12, 2:3] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1840 = stablehlo.reshape %1839 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1841 = stablehlo.multiply %1680, %1635 : tensor<256xf64>
        %cst_255 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1842 = stablehlo.broadcast_in_dim %cst_255, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1843 = stablehlo.add %1842, %1841 : tensor<256xf64>
        %1844 = stablehlo.multiply %1757, %1694 : tensor<256xf64>
        %1845 = stablehlo.add %1843, %1844 : tensor<256xf64>
        %1846 = stablehlo.subtract %1840, %1845 : tensor<256xf64>
        %1847 = stablehlo.divide %1846, %1766 : tensor<256xf64>
        %1848 = stablehlo.slice %1614 [0:256, 3:4, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1849 = stablehlo.reshape %1848 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1850 = stablehlo.multiply %1640, %1640 : tensor<256xf64>
        %cst_256 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1851 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1852 = stablehlo.add %1851, %1850 : tensor<256xf64>
        %1853 = stablehlo.multiply %1701, %1701 : tensor<256xf64>
        %1854 = stablehlo.add %1852, %1853 : tensor<256xf64>
        %1855 = stablehlo.multiply %1775, %1775 : tensor<256xf64>
        %1856 = stablehlo.add %1854, %1855 : tensor<256xf64>
        %1857 = stablehlo.subtract %1849, %1856 : tensor<256xf64>
        %1858 = stablehlo.sqrt %1857 : tensor<256xf64>
        %1859 = stablehlo.slice %1614 [0:256, 4:5, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1860 = stablehlo.reshape %1859 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1861 = stablehlo.multiply %1645, %1640 : tensor<256xf64>
        %cst_257 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1862 = stablehlo.broadcast_in_dim %cst_257, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1863 = stablehlo.add %1862, %1861 : tensor<256xf64>
        %1864 = stablehlo.multiply %1708, %1701 : tensor<256xf64>
        %1865 = stablehlo.add %1863, %1864 : tensor<256xf64>
        %1866 = stablehlo.multiply %1784, %1775 : tensor<256xf64>
        %1867 = stablehlo.add %1865, %1866 : tensor<256xf64>
        %1868 = stablehlo.subtract %1860, %1867 : tensor<256xf64>
        %1869 = stablehlo.divide %1868, %1858 : tensor<256xf64>
        %1870 = stablehlo.slice %1614 [0:256, 5:6, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1871 = stablehlo.reshape %1870 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1872 = stablehlo.multiply %1650, %1640 : tensor<256xf64>
        %cst_258 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1873 = stablehlo.broadcast_in_dim %cst_258, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1874 = stablehlo.add %1873, %1872 : tensor<256xf64>
        %1875 = stablehlo.multiply %1715, %1701 : tensor<256xf64>
        %1876 = stablehlo.add %1874, %1875 : tensor<256xf64>
        %1877 = stablehlo.multiply %1793, %1775 : tensor<256xf64>
        %1878 = stablehlo.add %1876, %1877 : tensor<256xf64>
        %1879 = stablehlo.subtract %1871, %1878 : tensor<256xf64>
        %1880 = stablehlo.divide %1879, %1858 : tensor<256xf64>
        %1881 = stablehlo.slice %1614 [0:256, 6:7, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1882 = stablehlo.reshape %1881 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1883 = stablehlo.multiply %1655, %1640 : tensor<256xf64>
        %cst_259 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1884 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1885 = stablehlo.add %1884, %1883 : tensor<256xf64>
        %1886 = stablehlo.multiply %1722, %1701 : tensor<256xf64>
        %1887 = stablehlo.add %1885, %1886 : tensor<256xf64>
        %1888 = stablehlo.multiply %1802, %1775 : tensor<256xf64>
        %1889 = stablehlo.add %1887, %1888 : tensor<256xf64>
        %1890 = stablehlo.subtract %1882, %1889 : tensor<256xf64>
        %1891 = stablehlo.divide %1890, %1858 : tensor<256xf64>
        %1892 = stablehlo.slice %1614 [0:256, 7:8, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1893 = stablehlo.reshape %1892 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1894 = stablehlo.multiply %1660, %1640 : tensor<256xf64>
        %cst_260 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1895 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1896 = stablehlo.add %1895, %1894 : tensor<256xf64>
        %1897 = stablehlo.multiply %1729, %1701 : tensor<256xf64>
        %1898 = stablehlo.add %1896, %1897 : tensor<256xf64>
        %1899 = stablehlo.multiply %1811, %1775 : tensor<256xf64>
        %1900 = stablehlo.add %1898, %1899 : tensor<256xf64>
        %1901 = stablehlo.subtract %1893, %1900 : tensor<256xf64>
        %1902 = stablehlo.divide %1901, %1858 : tensor<256xf64>
        %1903 = stablehlo.slice %1614 [0:256, 8:9, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1904 = stablehlo.reshape %1903 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1905 = stablehlo.multiply %1665, %1640 : tensor<256xf64>
        %cst_261 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1906 = stablehlo.broadcast_in_dim %cst_261, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1907 = stablehlo.add %1906, %1905 : tensor<256xf64>
        %1908 = stablehlo.multiply %1736, %1701 : tensor<256xf64>
        %1909 = stablehlo.add %1907, %1908 : tensor<256xf64>
        %1910 = stablehlo.multiply %1820, %1775 : tensor<256xf64>
        %1911 = stablehlo.add %1909, %1910 : tensor<256xf64>
        %1912 = stablehlo.subtract %1904, %1911 : tensor<256xf64>
        %1913 = stablehlo.divide %1912, %1858 : tensor<256xf64>
        %1914 = stablehlo.slice %1614 [0:256, 9:10, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1915 = stablehlo.reshape %1914 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1916 = stablehlo.multiply %1670, %1640 : tensor<256xf64>
        %cst_262 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1917 = stablehlo.broadcast_in_dim %cst_262, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1918 = stablehlo.add %1917, %1916 : tensor<256xf64>
        %1919 = stablehlo.multiply %1743, %1701 : tensor<256xf64>
        %1920 = stablehlo.add %1918, %1919 : tensor<256xf64>
        %1921 = stablehlo.multiply %1829, %1775 : tensor<256xf64>
        %1922 = stablehlo.add %1920, %1921 : tensor<256xf64>
        %1923 = stablehlo.subtract %1915, %1922 : tensor<256xf64>
        %1924 = stablehlo.divide %1923, %1858 : tensor<256xf64>
        %1925 = stablehlo.slice %1614 [0:256, 10:11, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1926 = stablehlo.reshape %1925 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1927 = stablehlo.multiply %1675, %1640 : tensor<256xf64>
        %cst_263 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1928 = stablehlo.broadcast_in_dim %cst_263, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1929 = stablehlo.add %1928, %1927 : tensor<256xf64>
        %1930 = stablehlo.multiply %1750, %1701 : tensor<256xf64>
        %1931 = stablehlo.add %1929, %1930 : tensor<256xf64>
        %1932 = stablehlo.multiply %1838, %1775 : tensor<256xf64>
        %1933 = stablehlo.add %1931, %1932 : tensor<256xf64>
        %1934 = stablehlo.subtract %1926, %1933 : tensor<256xf64>
        %1935 = stablehlo.divide %1934, %1858 : tensor<256xf64>
        %1936 = stablehlo.slice %1614 [0:256, 11:12, 3:4] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1937 = stablehlo.reshape %1936 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1938 = stablehlo.multiply %1680, %1640 : tensor<256xf64>
        %cst_264 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1939 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1940 = stablehlo.add %1939, %1938 : tensor<256xf64>
        %1941 = stablehlo.multiply %1757, %1701 : tensor<256xf64>
        %1942 = stablehlo.add %1940, %1941 : tensor<256xf64>
        %1943 = stablehlo.multiply %1847, %1775 : tensor<256xf64>
        %1944 = stablehlo.add %1942, %1943 : tensor<256xf64>
        %1945 = stablehlo.subtract %1937, %1944 : tensor<256xf64>
        %1946 = stablehlo.divide %1945, %1858 : tensor<256xf64>
        %1947 = stablehlo.slice %1614 [0:256, 4:5, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1948 = stablehlo.reshape %1947 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1949 = stablehlo.multiply %1645, %1645 : tensor<256xf64>
        %cst_265 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1950 = stablehlo.broadcast_in_dim %cst_265, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1951 = stablehlo.add %1950, %1949 : tensor<256xf64>
        %1952 = stablehlo.multiply %1708, %1708 : tensor<256xf64>
        %1953 = stablehlo.add %1951, %1952 : tensor<256xf64>
        %1954 = stablehlo.multiply %1784, %1784 : tensor<256xf64>
        %1955 = stablehlo.add %1953, %1954 : tensor<256xf64>
        %1956 = stablehlo.multiply %1869, %1869 : tensor<256xf64>
        %1957 = stablehlo.add %1955, %1956 : tensor<256xf64>
        %1958 = stablehlo.subtract %1948, %1957 : tensor<256xf64>
        %1959 = stablehlo.sqrt %1958 : tensor<256xf64>
        %1960 = stablehlo.slice %1614 [0:256, 5:6, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1961 = stablehlo.reshape %1960 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1962 = stablehlo.multiply %1650, %1645 : tensor<256xf64>
        %cst_266 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1963 = stablehlo.broadcast_in_dim %cst_266, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1964 = stablehlo.add %1963, %1962 : tensor<256xf64>
        %1965 = stablehlo.multiply %1715, %1708 : tensor<256xf64>
        %1966 = stablehlo.add %1964, %1965 : tensor<256xf64>
        %1967 = stablehlo.multiply %1793, %1784 : tensor<256xf64>
        %1968 = stablehlo.add %1966, %1967 : tensor<256xf64>
        %1969 = stablehlo.multiply %1880, %1869 : tensor<256xf64>
        %1970 = stablehlo.add %1968, %1969 : tensor<256xf64>
        %1971 = stablehlo.subtract %1961, %1970 : tensor<256xf64>
        %1972 = stablehlo.divide %1971, %1959 : tensor<256xf64>
        %1973 = stablehlo.slice %1614 [0:256, 6:7, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1974 = stablehlo.reshape %1973 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1975 = stablehlo.multiply %1655, %1645 : tensor<256xf64>
        %cst_267 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1976 = stablehlo.broadcast_in_dim %cst_267, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1977 = stablehlo.add %1976, %1975 : tensor<256xf64>
        %1978 = stablehlo.multiply %1722, %1708 : tensor<256xf64>
        %1979 = stablehlo.add %1977, %1978 : tensor<256xf64>
        %1980 = stablehlo.multiply %1802, %1784 : tensor<256xf64>
        %1981 = stablehlo.add %1979, %1980 : tensor<256xf64>
        %1982 = stablehlo.multiply %1891, %1869 : tensor<256xf64>
        %1983 = stablehlo.add %1981, %1982 : tensor<256xf64>
        %1984 = stablehlo.subtract %1974, %1983 : tensor<256xf64>
        %1985 = stablehlo.divide %1984, %1959 : tensor<256xf64>
        %1986 = stablehlo.slice %1614 [0:256, 7:8, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %1987 = stablehlo.reshape %1986 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %1988 = stablehlo.multiply %1660, %1645 : tensor<256xf64>
        %cst_268 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %1989 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %1990 = stablehlo.add %1989, %1988 : tensor<256xf64>
        %1991 = stablehlo.multiply %1729, %1708 : tensor<256xf64>
        %1992 = stablehlo.add %1990, %1991 : tensor<256xf64>
        %1993 = stablehlo.multiply %1811, %1784 : tensor<256xf64>
        %1994 = stablehlo.add %1992, %1993 : tensor<256xf64>
        %1995 = stablehlo.multiply %1902, %1869 : tensor<256xf64>
        %1996 = stablehlo.add %1994, %1995 : tensor<256xf64>
        %1997 = stablehlo.subtract %1987, %1996 : tensor<256xf64>
        %1998 = stablehlo.divide %1997, %1959 : tensor<256xf64>
        %1999 = stablehlo.slice %1614 [0:256, 8:9, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2000 = stablehlo.reshape %1999 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2001 = stablehlo.multiply %1665, %1645 : tensor<256xf64>
        %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2002 = stablehlo.broadcast_in_dim %cst_269, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2003 = stablehlo.add %2002, %2001 : tensor<256xf64>
        %2004 = stablehlo.multiply %1736, %1708 : tensor<256xf64>
        %2005 = stablehlo.add %2003, %2004 : tensor<256xf64>
        %2006 = stablehlo.multiply %1820, %1784 : tensor<256xf64>
        %2007 = stablehlo.add %2005, %2006 : tensor<256xf64>
        %2008 = stablehlo.multiply %1913, %1869 : tensor<256xf64>
        %2009 = stablehlo.add %2007, %2008 : tensor<256xf64>
        %2010 = stablehlo.subtract %2000, %2009 : tensor<256xf64>
        %2011 = stablehlo.divide %2010, %1959 : tensor<256xf64>
        %2012 = stablehlo.slice %1614 [0:256, 9:10, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2013 = stablehlo.reshape %2012 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2014 = stablehlo.multiply %1670, %1645 : tensor<256xf64>
        %cst_270 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2015 = stablehlo.broadcast_in_dim %cst_270, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2016 = stablehlo.add %2015, %2014 : tensor<256xf64>
        %2017 = stablehlo.multiply %1743, %1708 : tensor<256xf64>
        %2018 = stablehlo.add %2016, %2017 : tensor<256xf64>
        %2019 = stablehlo.multiply %1829, %1784 : tensor<256xf64>
        %2020 = stablehlo.add %2018, %2019 : tensor<256xf64>
        %2021 = stablehlo.multiply %1924, %1869 : tensor<256xf64>
        %2022 = stablehlo.add %2020, %2021 : tensor<256xf64>
        %2023 = stablehlo.subtract %2013, %2022 : tensor<256xf64>
        %2024 = stablehlo.divide %2023, %1959 : tensor<256xf64>
        %2025 = stablehlo.slice %1614 [0:256, 10:11, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2026 = stablehlo.reshape %2025 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2027 = stablehlo.multiply %1675, %1645 : tensor<256xf64>
        %cst_271 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2028 = stablehlo.broadcast_in_dim %cst_271, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2029 = stablehlo.add %2028, %2027 : tensor<256xf64>
        %2030 = stablehlo.multiply %1750, %1708 : tensor<256xf64>
        %2031 = stablehlo.add %2029, %2030 : tensor<256xf64>
        %2032 = stablehlo.multiply %1838, %1784 : tensor<256xf64>
        %2033 = stablehlo.add %2031, %2032 : tensor<256xf64>
        %2034 = stablehlo.multiply %1935, %1869 : tensor<256xf64>
        %2035 = stablehlo.add %2033, %2034 : tensor<256xf64>
        %2036 = stablehlo.subtract %2026, %2035 : tensor<256xf64>
        %2037 = stablehlo.divide %2036, %1959 : tensor<256xf64>
        %2038 = stablehlo.slice %1614 [0:256, 11:12, 4:5] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2039 = stablehlo.reshape %2038 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2040 = stablehlo.multiply %1680, %1645 : tensor<256xf64>
        %cst_272 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2041 = stablehlo.broadcast_in_dim %cst_272, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2042 = stablehlo.add %2041, %2040 : tensor<256xf64>
        %2043 = stablehlo.multiply %1757, %1708 : tensor<256xf64>
        %2044 = stablehlo.add %2042, %2043 : tensor<256xf64>
        %2045 = stablehlo.multiply %1847, %1784 : tensor<256xf64>
        %2046 = stablehlo.add %2044, %2045 : tensor<256xf64>
        %2047 = stablehlo.multiply %1946, %1869 : tensor<256xf64>
        %2048 = stablehlo.add %2046, %2047 : tensor<256xf64>
        %2049 = stablehlo.subtract %2039, %2048 : tensor<256xf64>
        %2050 = stablehlo.divide %2049, %1959 : tensor<256xf64>
        %2051 = stablehlo.slice %1614 [0:256, 5:6, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2052 = stablehlo.reshape %2051 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2053 = stablehlo.multiply %1650, %1650 : tensor<256xf64>
        %cst_273 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2054 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2055 = stablehlo.add %2054, %2053 : tensor<256xf64>
        %2056 = stablehlo.multiply %1715, %1715 : tensor<256xf64>
        %2057 = stablehlo.add %2055, %2056 : tensor<256xf64>
        %2058 = stablehlo.multiply %1793, %1793 : tensor<256xf64>
        %2059 = stablehlo.add %2057, %2058 : tensor<256xf64>
        %2060 = stablehlo.multiply %1880, %1880 : tensor<256xf64>
        %2061 = stablehlo.add %2059, %2060 : tensor<256xf64>
        %2062 = stablehlo.multiply %1972, %1972 : tensor<256xf64>
        %2063 = stablehlo.add %2061, %2062 : tensor<256xf64>
        %2064 = stablehlo.subtract %2052, %2063 : tensor<256xf64>
        %2065 = stablehlo.sqrt %2064 : tensor<256xf64>
        %2066 = stablehlo.slice %1614 [0:256, 6:7, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2067 = stablehlo.reshape %2066 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2068 = stablehlo.multiply %1655, %1650 : tensor<256xf64>
        %cst_274 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2069 = stablehlo.broadcast_in_dim %cst_274, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2070 = stablehlo.add %2069, %2068 : tensor<256xf64>
        %2071 = stablehlo.multiply %1722, %1715 : tensor<256xf64>
        %2072 = stablehlo.add %2070, %2071 : tensor<256xf64>
        %2073 = stablehlo.multiply %1802, %1793 : tensor<256xf64>
        %2074 = stablehlo.add %2072, %2073 : tensor<256xf64>
        %2075 = stablehlo.multiply %1891, %1880 : tensor<256xf64>
        %2076 = stablehlo.add %2074, %2075 : tensor<256xf64>
        %2077 = stablehlo.multiply %1985, %1972 : tensor<256xf64>
        %2078 = stablehlo.add %2076, %2077 : tensor<256xf64>
        %2079 = stablehlo.subtract %2067, %2078 : tensor<256xf64>
        %2080 = stablehlo.divide %2079, %2065 : tensor<256xf64>
        %2081 = stablehlo.slice %1614 [0:256, 7:8, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2082 = stablehlo.reshape %2081 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2083 = stablehlo.multiply %1660, %1650 : tensor<256xf64>
        %cst_275 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2084 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2085 = stablehlo.add %2084, %2083 : tensor<256xf64>
        %2086 = stablehlo.multiply %1729, %1715 : tensor<256xf64>
        %2087 = stablehlo.add %2085, %2086 : tensor<256xf64>
        %2088 = stablehlo.multiply %1811, %1793 : tensor<256xf64>
        %2089 = stablehlo.add %2087, %2088 : tensor<256xf64>
        %2090 = stablehlo.multiply %1902, %1880 : tensor<256xf64>
        %2091 = stablehlo.add %2089, %2090 : tensor<256xf64>
        %2092 = stablehlo.multiply %1998, %1972 : tensor<256xf64>
        %2093 = stablehlo.add %2091, %2092 : tensor<256xf64>
        %2094 = stablehlo.subtract %2082, %2093 : tensor<256xf64>
        %2095 = stablehlo.divide %2094, %2065 : tensor<256xf64>
        %2096 = stablehlo.slice %1614 [0:256, 8:9, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2097 = stablehlo.reshape %2096 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2098 = stablehlo.multiply %1665, %1650 : tensor<256xf64>
        %cst_276 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2099 = stablehlo.broadcast_in_dim %cst_276, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2100 = stablehlo.add %2099, %2098 : tensor<256xf64>
        %2101 = stablehlo.multiply %1736, %1715 : tensor<256xf64>
        %2102 = stablehlo.add %2100, %2101 : tensor<256xf64>
        %2103 = stablehlo.multiply %1820, %1793 : tensor<256xf64>
        %2104 = stablehlo.add %2102, %2103 : tensor<256xf64>
        %2105 = stablehlo.multiply %1913, %1880 : tensor<256xf64>
        %2106 = stablehlo.add %2104, %2105 : tensor<256xf64>
        %2107 = stablehlo.multiply %2011, %1972 : tensor<256xf64>
        %2108 = stablehlo.add %2106, %2107 : tensor<256xf64>
        %2109 = stablehlo.subtract %2097, %2108 : tensor<256xf64>
        %2110 = stablehlo.divide %2109, %2065 : tensor<256xf64>
        %2111 = stablehlo.slice %1614 [0:256, 9:10, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2112 = stablehlo.reshape %2111 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2113 = stablehlo.multiply %1670, %1650 : tensor<256xf64>
        %cst_277 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2114 = stablehlo.broadcast_in_dim %cst_277, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2115 = stablehlo.add %2114, %2113 : tensor<256xf64>
        %2116 = stablehlo.multiply %1743, %1715 : tensor<256xf64>
        %2117 = stablehlo.add %2115, %2116 : tensor<256xf64>
        %2118 = stablehlo.multiply %1829, %1793 : tensor<256xf64>
        %2119 = stablehlo.add %2117, %2118 : tensor<256xf64>
        %2120 = stablehlo.multiply %1924, %1880 : tensor<256xf64>
        %2121 = stablehlo.add %2119, %2120 : tensor<256xf64>
        %2122 = stablehlo.multiply %2024, %1972 : tensor<256xf64>
        %2123 = stablehlo.add %2121, %2122 : tensor<256xf64>
        %2124 = stablehlo.subtract %2112, %2123 : tensor<256xf64>
        %2125 = stablehlo.divide %2124, %2065 : tensor<256xf64>
        %2126 = stablehlo.slice %1614 [0:256, 10:11, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2127 = stablehlo.reshape %2126 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2128 = stablehlo.multiply %1675, %1650 : tensor<256xf64>
        %cst_278 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2129 = stablehlo.broadcast_in_dim %cst_278, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2130 = stablehlo.add %2129, %2128 : tensor<256xf64>
        %2131 = stablehlo.multiply %1750, %1715 : tensor<256xf64>
        %2132 = stablehlo.add %2130, %2131 : tensor<256xf64>
        %2133 = stablehlo.multiply %1838, %1793 : tensor<256xf64>
        %2134 = stablehlo.add %2132, %2133 : tensor<256xf64>
        %2135 = stablehlo.multiply %1935, %1880 : tensor<256xf64>
        %2136 = stablehlo.add %2134, %2135 : tensor<256xf64>
        %2137 = stablehlo.multiply %2037, %1972 : tensor<256xf64>
        %2138 = stablehlo.add %2136, %2137 : tensor<256xf64>
        %2139 = stablehlo.subtract %2127, %2138 : tensor<256xf64>
        %2140 = stablehlo.divide %2139, %2065 : tensor<256xf64>
        %2141 = stablehlo.slice %1614 [0:256, 11:12, 5:6] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2142 = stablehlo.reshape %2141 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2143 = stablehlo.multiply %1680, %1650 : tensor<256xf64>
        %cst_279 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2144 = stablehlo.broadcast_in_dim %cst_279, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2145 = stablehlo.add %2144, %2143 : tensor<256xf64>
        %2146 = stablehlo.multiply %1757, %1715 : tensor<256xf64>
        %2147 = stablehlo.add %2145, %2146 : tensor<256xf64>
        %2148 = stablehlo.multiply %1847, %1793 : tensor<256xf64>
        %2149 = stablehlo.add %2147, %2148 : tensor<256xf64>
        %2150 = stablehlo.multiply %1946, %1880 : tensor<256xf64>
        %2151 = stablehlo.add %2149, %2150 : tensor<256xf64>
        %2152 = stablehlo.multiply %2050, %1972 : tensor<256xf64>
        %2153 = stablehlo.add %2151, %2152 : tensor<256xf64>
        %2154 = stablehlo.subtract %2142, %2153 : tensor<256xf64>
        %2155 = stablehlo.divide %2154, %2065 : tensor<256xf64>
        %2156 = stablehlo.slice %1614 [0:256, 6:7, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2157 = stablehlo.reshape %2156 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2158 = stablehlo.multiply %1655, %1655 : tensor<256xf64>
        %cst_280 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2159 = stablehlo.broadcast_in_dim %cst_280, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2160 = stablehlo.add %2159, %2158 : tensor<256xf64>
        %2161 = stablehlo.multiply %1722, %1722 : tensor<256xf64>
        %2162 = stablehlo.add %2160, %2161 : tensor<256xf64>
        %2163 = stablehlo.multiply %1802, %1802 : tensor<256xf64>
        %2164 = stablehlo.add %2162, %2163 : tensor<256xf64>
        %2165 = stablehlo.multiply %1891, %1891 : tensor<256xf64>
        %2166 = stablehlo.add %2164, %2165 : tensor<256xf64>
        %2167 = stablehlo.multiply %1985, %1985 : tensor<256xf64>
        %2168 = stablehlo.add %2166, %2167 : tensor<256xf64>
        %2169 = stablehlo.multiply %2080, %2080 : tensor<256xf64>
        %2170 = stablehlo.add %2168, %2169 : tensor<256xf64>
        %2171 = stablehlo.subtract %2157, %2170 : tensor<256xf64>
        %2172 = stablehlo.sqrt %2171 : tensor<256xf64>
        %2173 = stablehlo.slice %1614 [0:256, 7:8, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2174 = stablehlo.reshape %2173 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2175 = stablehlo.multiply %1660, %1655 : tensor<256xf64>
        %cst_281 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2176 = stablehlo.broadcast_in_dim %cst_281, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2177 = stablehlo.add %2176, %2175 : tensor<256xf64>
        %2178 = stablehlo.multiply %1729, %1722 : tensor<256xf64>
        %2179 = stablehlo.add %2177, %2178 : tensor<256xf64>
        %2180 = stablehlo.multiply %1811, %1802 : tensor<256xf64>
        %2181 = stablehlo.add %2179, %2180 : tensor<256xf64>
        %2182 = stablehlo.multiply %1902, %1891 : tensor<256xf64>
        %2183 = stablehlo.add %2181, %2182 : tensor<256xf64>
        %2184 = stablehlo.multiply %1998, %1985 : tensor<256xf64>
        %2185 = stablehlo.add %2183, %2184 : tensor<256xf64>
        %2186 = stablehlo.multiply %2095, %2080 : tensor<256xf64>
        %2187 = stablehlo.add %2185, %2186 : tensor<256xf64>
        %2188 = stablehlo.subtract %2174, %2187 : tensor<256xf64>
        %2189 = stablehlo.divide %2188, %2172 : tensor<256xf64>
        %2190 = stablehlo.slice %1614 [0:256, 8:9, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2191 = stablehlo.reshape %2190 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2192 = stablehlo.multiply %1665, %1655 : tensor<256xf64>
        %cst_282 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2193 = stablehlo.broadcast_in_dim %cst_282, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2194 = stablehlo.add %2193, %2192 : tensor<256xf64>
        %2195 = stablehlo.multiply %1736, %1722 : tensor<256xf64>
        %2196 = stablehlo.add %2194, %2195 : tensor<256xf64>
        %2197 = stablehlo.multiply %1820, %1802 : tensor<256xf64>
        %2198 = stablehlo.add %2196, %2197 : tensor<256xf64>
        %2199 = stablehlo.multiply %1913, %1891 : tensor<256xf64>
        %2200 = stablehlo.add %2198, %2199 : tensor<256xf64>
        %2201 = stablehlo.multiply %2011, %1985 : tensor<256xf64>
        %2202 = stablehlo.add %2200, %2201 : tensor<256xf64>
        %2203 = stablehlo.multiply %2110, %2080 : tensor<256xf64>
        %2204 = stablehlo.add %2202, %2203 : tensor<256xf64>
        %2205 = stablehlo.subtract %2191, %2204 : tensor<256xf64>
        %2206 = stablehlo.divide %2205, %2172 : tensor<256xf64>
        %2207 = stablehlo.slice %1614 [0:256, 9:10, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2208 = stablehlo.reshape %2207 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2209 = stablehlo.multiply %1670, %1655 : tensor<256xf64>
        %cst_283 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2210 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2211 = stablehlo.add %2210, %2209 : tensor<256xf64>
        %2212 = stablehlo.multiply %1743, %1722 : tensor<256xf64>
        %2213 = stablehlo.add %2211, %2212 : tensor<256xf64>
        %2214 = stablehlo.multiply %1829, %1802 : tensor<256xf64>
        %2215 = stablehlo.add %2213, %2214 : tensor<256xf64>
        %2216 = stablehlo.multiply %1924, %1891 : tensor<256xf64>
        %2217 = stablehlo.add %2215, %2216 : tensor<256xf64>
        %2218 = stablehlo.multiply %2024, %1985 : tensor<256xf64>
        %2219 = stablehlo.add %2217, %2218 : tensor<256xf64>
        %2220 = stablehlo.multiply %2125, %2080 : tensor<256xf64>
        %2221 = stablehlo.add %2219, %2220 : tensor<256xf64>
        %2222 = stablehlo.subtract %2208, %2221 : tensor<256xf64>
        %2223 = stablehlo.divide %2222, %2172 : tensor<256xf64>
        %2224 = stablehlo.slice %1614 [0:256, 10:11, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2225 = stablehlo.reshape %2224 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2226 = stablehlo.multiply %1675, %1655 : tensor<256xf64>
        %cst_284 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2227 = stablehlo.broadcast_in_dim %cst_284, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2228 = stablehlo.add %2227, %2226 : tensor<256xf64>
        %2229 = stablehlo.multiply %1750, %1722 : tensor<256xf64>
        %2230 = stablehlo.add %2228, %2229 : tensor<256xf64>
        %2231 = stablehlo.multiply %1838, %1802 : tensor<256xf64>
        %2232 = stablehlo.add %2230, %2231 : tensor<256xf64>
        %2233 = stablehlo.multiply %1935, %1891 : tensor<256xf64>
        %2234 = stablehlo.add %2232, %2233 : tensor<256xf64>
        %2235 = stablehlo.multiply %2037, %1985 : tensor<256xf64>
        %2236 = stablehlo.add %2234, %2235 : tensor<256xf64>
        %2237 = stablehlo.multiply %2140, %2080 : tensor<256xf64>
        %2238 = stablehlo.add %2236, %2237 : tensor<256xf64>
        %2239 = stablehlo.subtract %2225, %2238 : tensor<256xf64>
        %2240 = stablehlo.divide %2239, %2172 : tensor<256xf64>
        %2241 = stablehlo.slice %1614 [0:256, 11:12, 6:7] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2242 = stablehlo.reshape %2241 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2243 = stablehlo.multiply %1680, %1655 : tensor<256xf64>
        %cst_285 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2244 = stablehlo.broadcast_in_dim %cst_285, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2245 = stablehlo.add %2244, %2243 : tensor<256xf64>
        %2246 = stablehlo.multiply %1757, %1722 : tensor<256xf64>
        %2247 = stablehlo.add %2245, %2246 : tensor<256xf64>
        %2248 = stablehlo.multiply %1847, %1802 : tensor<256xf64>
        %2249 = stablehlo.add %2247, %2248 : tensor<256xf64>
        %2250 = stablehlo.multiply %1946, %1891 : tensor<256xf64>
        %2251 = stablehlo.add %2249, %2250 : tensor<256xf64>
        %2252 = stablehlo.multiply %2050, %1985 : tensor<256xf64>
        %2253 = stablehlo.add %2251, %2252 : tensor<256xf64>
        %2254 = stablehlo.multiply %2155, %2080 : tensor<256xf64>
        %2255 = stablehlo.add %2253, %2254 : tensor<256xf64>
        %2256 = stablehlo.subtract %2242, %2255 : tensor<256xf64>
        %2257 = stablehlo.divide %2256, %2172 : tensor<256xf64>
        %2258 = stablehlo.slice %1614 [0:256, 7:8, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2259 = stablehlo.reshape %2258 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2260 = stablehlo.multiply %1660, %1660 : tensor<256xf64>
        %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2261 = stablehlo.broadcast_in_dim %cst_286, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2262 = stablehlo.add %2261, %2260 : tensor<256xf64>
        %2263 = stablehlo.multiply %1729, %1729 : tensor<256xf64>
        %2264 = stablehlo.add %2262, %2263 : tensor<256xf64>
        %2265 = stablehlo.multiply %1811, %1811 : tensor<256xf64>
        %2266 = stablehlo.add %2264, %2265 : tensor<256xf64>
        %2267 = stablehlo.multiply %1902, %1902 : tensor<256xf64>
        %2268 = stablehlo.add %2266, %2267 : tensor<256xf64>
        %2269 = stablehlo.multiply %1998, %1998 : tensor<256xf64>
        %2270 = stablehlo.add %2268, %2269 : tensor<256xf64>
        %2271 = stablehlo.multiply %2095, %2095 : tensor<256xf64>
        %2272 = stablehlo.add %2270, %2271 : tensor<256xf64>
        %2273 = stablehlo.multiply %2189, %2189 : tensor<256xf64>
        %2274 = stablehlo.add %2272, %2273 : tensor<256xf64>
        %2275 = stablehlo.subtract %2259, %2274 : tensor<256xf64>
        %2276 = stablehlo.sqrt %2275 : tensor<256xf64>
        %2277 = stablehlo.slice %1614 [0:256, 8:9, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2278 = stablehlo.reshape %2277 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2279 = stablehlo.multiply %1665, %1660 : tensor<256xf64>
        %cst_287 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2280 = stablehlo.broadcast_in_dim %cst_287, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2281 = stablehlo.add %2280, %2279 : tensor<256xf64>
        %2282 = stablehlo.multiply %1736, %1729 : tensor<256xf64>
        %2283 = stablehlo.add %2281, %2282 : tensor<256xf64>
        %2284 = stablehlo.multiply %1820, %1811 : tensor<256xf64>
        %2285 = stablehlo.add %2283, %2284 : tensor<256xf64>
        %2286 = stablehlo.multiply %1913, %1902 : tensor<256xf64>
        %2287 = stablehlo.add %2285, %2286 : tensor<256xf64>
        %2288 = stablehlo.multiply %2011, %1998 : tensor<256xf64>
        %2289 = stablehlo.add %2287, %2288 : tensor<256xf64>
        %2290 = stablehlo.multiply %2110, %2095 : tensor<256xf64>
        %2291 = stablehlo.add %2289, %2290 : tensor<256xf64>
        %2292 = stablehlo.multiply %2206, %2189 : tensor<256xf64>
        %2293 = stablehlo.add %2291, %2292 : tensor<256xf64>
        %2294 = stablehlo.subtract %2278, %2293 : tensor<256xf64>
        %2295 = stablehlo.divide %2294, %2276 : tensor<256xf64>
        %2296 = stablehlo.slice %1614 [0:256, 9:10, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2297 = stablehlo.reshape %2296 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2298 = stablehlo.multiply %1670, %1660 : tensor<256xf64>
        %cst_288 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2299 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2300 = stablehlo.add %2299, %2298 : tensor<256xf64>
        %2301 = stablehlo.multiply %1743, %1729 : tensor<256xf64>
        %2302 = stablehlo.add %2300, %2301 : tensor<256xf64>
        %2303 = stablehlo.multiply %1829, %1811 : tensor<256xf64>
        %2304 = stablehlo.add %2302, %2303 : tensor<256xf64>
        %2305 = stablehlo.multiply %1924, %1902 : tensor<256xf64>
        %2306 = stablehlo.add %2304, %2305 : tensor<256xf64>
        %2307 = stablehlo.multiply %2024, %1998 : tensor<256xf64>
        %2308 = stablehlo.add %2306, %2307 : tensor<256xf64>
        %2309 = stablehlo.multiply %2125, %2095 : tensor<256xf64>
        %2310 = stablehlo.add %2308, %2309 : tensor<256xf64>
        %2311 = stablehlo.multiply %2223, %2189 : tensor<256xf64>
        %2312 = stablehlo.add %2310, %2311 : tensor<256xf64>
        %2313 = stablehlo.subtract %2297, %2312 : tensor<256xf64>
        %2314 = stablehlo.divide %2313, %2276 : tensor<256xf64>
        %2315 = stablehlo.slice %1614 [0:256, 10:11, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2316 = stablehlo.reshape %2315 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2317 = stablehlo.multiply %1675, %1660 : tensor<256xf64>
        %cst_289 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2318 = stablehlo.broadcast_in_dim %cst_289, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2319 = stablehlo.add %2318, %2317 : tensor<256xf64>
        %2320 = stablehlo.multiply %1750, %1729 : tensor<256xf64>
        %2321 = stablehlo.add %2319, %2320 : tensor<256xf64>
        %2322 = stablehlo.multiply %1838, %1811 : tensor<256xf64>
        %2323 = stablehlo.add %2321, %2322 : tensor<256xf64>
        %2324 = stablehlo.multiply %1935, %1902 : tensor<256xf64>
        %2325 = stablehlo.add %2323, %2324 : tensor<256xf64>
        %2326 = stablehlo.multiply %2037, %1998 : tensor<256xf64>
        %2327 = stablehlo.add %2325, %2326 : tensor<256xf64>
        %2328 = stablehlo.multiply %2140, %2095 : tensor<256xf64>
        %2329 = stablehlo.add %2327, %2328 : tensor<256xf64>
        %2330 = stablehlo.multiply %2240, %2189 : tensor<256xf64>
        %2331 = stablehlo.add %2329, %2330 : tensor<256xf64>
        %2332 = stablehlo.subtract %2316, %2331 : tensor<256xf64>
        %2333 = stablehlo.divide %2332, %2276 : tensor<256xf64>
        %2334 = stablehlo.slice %1614 [0:256, 11:12, 7:8] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2335 = stablehlo.reshape %2334 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2336 = stablehlo.multiply %1680, %1660 : tensor<256xf64>
        %cst_290 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2337 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2338 = stablehlo.add %2337, %2336 : tensor<256xf64>
        %2339 = stablehlo.multiply %1757, %1729 : tensor<256xf64>
        %2340 = stablehlo.add %2338, %2339 : tensor<256xf64>
        %2341 = stablehlo.multiply %1847, %1811 : tensor<256xf64>
        %2342 = stablehlo.add %2340, %2341 : tensor<256xf64>
        %2343 = stablehlo.multiply %1946, %1902 : tensor<256xf64>
        %2344 = stablehlo.add %2342, %2343 : tensor<256xf64>
        %2345 = stablehlo.multiply %2050, %1998 : tensor<256xf64>
        %2346 = stablehlo.add %2344, %2345 : tensor<256xf64>
        %2347 = stablehlo.multiply %2155, %2095 : tensor<256xf64>
        %2348 = stablehlo.add %2346, %2347 : tensor<256xf64>
        %2349 = stablehlo.multiply %2257, %2189 : tensor<256xf64>
        %2350 = stablehlo.add %2348, %2349 : tensor<256xf64>
        %2351 = stablehlo.subtract %2335, %2350 : tensor<256xf64>
        %2352 = stablehlo.divide %2351, %2276 : tensor<256xf64>
        %2353 = stablehlo.slice %1614 [0:256, 8:9, 8:9] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2354 = stablehlo.reshape %2353 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2355 = stablehlo.multiply %1665, %1665 : tensor<256xf64>
        %cst_291 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2356 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2357 = stablehlo.add %2356, %2355 : tensor<256xf64>
        %2358 = stablehlo.multiply %1736, %1736 : tensor<256xf64>
        %2359 = stablehlo.add %2357, %2358 : tensor<256xf64>
        %2360 = stablehlo.multiply %1820, %1820 : tensor<256xf64>
        %2361 = stablehlo.add %2359, %2360 : tensor<256xf64>
        %2362 = stablehlo.multiply %1913, %1913 : tensor<256xf64>
        %2363 = stablehlo.add %2361, %2362 : tensor<256xf64>
        %2364 = stablehlo.multiply %2011, %2011 : tensor<256xf64>
        %2365 = stablehlo.add %2363, %2364 : tensor<256xf64>
        %2366 = stablehlo.multiply %2110, %2110 : tensor<256xf64>
        %2367 = stablehlo.add %2365, %2366 : tensor<256xf64>
        %2368 = stablehlo.multiply %2206, %2206 : tensor<256xf64>
        %2369 = stablehlo.add %2367, %2368 : tensor<256xf64>
        %2370 = stablehlo.multiply %2295, %2295 : tensor<256xf64>
        %2371 = stablehlo.add %2369, %2370 : tensor<256xf64>
        %2372 = stablehlo.subtract %2354, %2371 : tensor<256xf64>
        %2373 = stablehlo.sqrt %2372 : tensor<256xf64>
        %2374 = stablehlo.slice %1614 [0:256, 9:10, 8:9] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2375 = stablehlo.reshape %2374 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2376 = stablehlo.multiply %1670, %1665 : tensor<256xf64>
        %cst_292 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2377 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2378 = stablehlo.add %2377, %2376 : tensor<256xf64>
        %2379 = stablehlo.multiply %1743, %1736 : tensor<256xf64>
        %2380 = stablehlo.add %2378, %2379 : tensor<256xf64>
        %2381 = stablehlo.multiply %1829, %1820 : tensor<256xf64>
        %2382 = stablehlo.add %2380, %2381 : tensor<256xf64>
        %2383 = stablehlo.multiply %1924, %1913 : tensor<256xf64>
        %2384 = stablehlo.add %2382, %2383 : tensor<256xf64>
        %2385 = stablehlo.multiply %2024, %2011 : tensor<256xf64>
        %2386 = stablehlo.add %2384, %2385 : tensor<256xf64>
        %2387 = stablehlo.multiply %2125, %2110 : tensor<256xf64>
        %2388 = stablehlo.add %2386, %2387 : tensor<256xf64>
        %2389 = stablehlo.multiply %2223, %2206 : tensor<256xf64>
        %2390 = stablehlo.add %2388, %2389 : tensor<256xf64>
        %2391 = stablehlo.multiply %2314, %2295 : tensor<256xf64>
        %2392 = stablehlo.add %2390, %2391 : tensor<256xf64>
        %2393 = stablehlo.subtract %2375, %2392 : tensor<256xf64>
        %2394 = stablehlo.divide %2393, %2373 : tensor<256xf64>
        %2395 = stablehlo.slice %1614 [0:256, 10:11, 8:9] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2396 = stablehlo.reshape %2395 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2397 = stablehlo.multiply %1675, %1665 : tensor<256xf64>
        %cst_293 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2398 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2399 = stablehlo.add %2398, %2397 : tensor<256xf64>
        %2400 = stablehlo.multiply %1750, %1736 : tensor<256xf64>
        %2401 = stablehlo.add %2399, %2400 : tensor<256xf64>
        %2402 = stablehlo.multiply %1838, %1820 : tensor<256xf64>
        %2403 = stablehlo.add %2401, %2402 : tensor<256xf64>
        %2404 = stablehlo.multiply %1935, %1913 : tensor<256xf64>
        %2405 = stablehlo.add %2403, %2404 : tensor<256xf64>
        %2406 = stablehlo.multiply %2037, %2011 : tensor<256xf64>
        %2407 = stablehlo.add %2405, %2406 : tensor<256xf64>
        %2408 = stablehlo.multiply %2140, %2110 : tensor<256xf64>
        %2409 = stablehlo.add %2407, %2408 : tensor<256xf64>
        %2410 = stablehlo.multiply %2240, %2206 : tensor<256xf64>
        %2411 = stablehlo.add %2409, %2410 : tensor<256xf64>
        %2412 = stablehlo.multiply %2333, %2295 : tensor<256xf64>
        %2413 = stablehlo.add %2411, %2412 : tensor<256xf64>
        %2414 = stablehlo.subtract %2396, %2413 : tensor<256xf64>
        %2415 = stablehlo.divide %2414, %2373 : tensor<256xf64>
        %2416 = stablehlo.slice %1614 [0:256, 11:12, 8:9] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2417 = stablehlo.reshape %2416 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2418 = stablehlo.multiply %1680, %1665 : tensor<256xf64>
        %cst_294 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2419 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2420 = stablehlo.add %2419, %2418 : tensor<256xf64>
        %2421 = stablehlo.multiply %1757, %1736 : tensor<256xf64>
        %2422 = stablehlo.add %2420, %2421 : tensor<256xf64>
        %2423 = stablehlo.multiply %1847, %1820 : tensor<256xf64>
        %2424 = stablehlo.add %2422, %2423 : tensor<256xf64>
        %2425 = stablehlo.multiply %1946, %1913 : tensor<256xf64>
        %2426 = stablehlo.add %2424, %2425 : tensor<256xf64>
        %2427 = stablehlo.multiply %2050, %2011 : tensor<256xf64>
        %2428 = stablehlo.add %2426, %2427 : tensor<256xf64>
        %2429 = stablehlo.multiply %2155, %2110 : tensor<256xf64>
        %2430 = stablehlo.add %2428, %2429 : tensor<256xf64>
        %2431 = stablehlo.multiply %2257, %2206 : tensor<256xf64>
        %2432 = stablehlo.add %2430, %2431 : tensor<256xf64>
        %2433 = stablehlo.multiply %2352, %2295 : tensor<256xf64>
        %2434 = stablehlo.add %2432, %2433 : tensor<256xf64>
        %2435 = stablehlo.subtract %2417, %2434 : tensor<256xf64>
        %2436 = stablehlo.divide %2435, %2373 : tensor<256xf64>
        %2437 = stablehlo.slice %1614 [0:256, 9:10, 9:10] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2438 = stablehlo.reshape %2437 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2439 = stablehlo.multiply %1670, %1670 : tensor<256xf64>
        %cst_295 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2440 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2441 = stablehlo.add %2440, %2439 : tensor<256xf64>
        %2442 = stablehlo.multiply %1743, %1743 : tensor<256xf64>
        %2443 = stablehlo.add %2441, %2442 : tensor<256xf64>
        %2444 = stablehlo.multiply %1829, %1829 : tensor<256xf64>
        %2445 = stablehlo.add %2443, %2444 : tensor<256xf64>
        %2446 = stablehlo.multiply %1924, %1924 : tensor<256xf64>
        %2447 = stablehlo.add %2445, %2446 : tensor<256xf64>
        %2448 = stablehlo.multiply %2024, %2024 : tensor<256xf64>
        %2449 = stablehlo.add %2447, %2448 : tensor<256xf64>
        %2450 = stablehlo.multiply %2125, %2125 : tensor<256xf64>
        %2451 = stablehlo.add %2449, %2450 : tensor<256xf64>
        %2452 = stablehlo.multiply %2223, %2223 : tensor<256xf64>
        %2453 = stablehlo.add %2451, %2452 : tensor<256xf64>
        %2454 = stablehlo.multiply %2314, %2314 : tensor<256xf64>
        %2455 = stablehlo.add %2453, %2454 : tensor<256xf64>
        %2456 = stablehlo.multiply %2394, %2394 : tensor<256xf64>
        %2457 = stablehlo.add %2455, %2456 : tensor<256xf64>
        %2458 = stablehlo.subtract %2438, %2457 : tensor<256xf64>
        %2459 = stablehlo.sqrt %2458 : tensor<256xf64>
        %2460 = stablehlo.slice %1614 [0:256, 10:11, 9:10] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2461 = stablehlo.reshape %2460 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2462 = stablehlo.multiply %1675, %1670 : tensor<256xf64>
        %cst_296 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2463 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2464 = stablehlo.add %2463, %2462 : tensor<256xf64>
        %2465 = stablehlo.multiply %1750, %1743 : tensor<256xf64>
        %2466 = stablehlo.add %2464, %2465 : tensor<256xf64>
        %2467 = stablehlo.multiply %1838, %1829 : tensor<256xf64>
        %2468 = stablehlo.add %2466, %2467 : tensor<256xf64>
        %2469 = stablehlo.multiply %1935, %1924 : tensor<256xf64>
        %2470 = stablehlo.add %2468, %2469 : tensor<256xf64>
        %2471 = stablehlo.multiply %2037, %2024 : tensor<256xf64>
        %2472 = stablehlo.add %2470, %2471 : tensor<256xf64>
        %2473 = stablehlo.multiply %2140, %2125 : tensor<256xf64>
        %2474 = stablehlo.add %2472, %2473 : tensor<256xf64>
        %2475 = stablehlo.multiply %2240, %2223 : tensor<256xf64>
        %2476 = stablehlo.add %2474, %2475 : tensor<256xf64>
        %2477 = stablehlo.multiply %2333, %2314 : tensor<256xf64>
        %2478 = stablehlo.add %2476, %2477 : tensor<256xf64>
        %2479 = stablehlo.multiply %2415, %2394 : tensor<256xf64>
        %2480 = stablehlo.add %2478, %2479 : tensor<256xf64>
        %2481 = stablehlo.subtract %2461, %2480 : tensor<256xf64>
        %2482 = stablehlo.divide %2481, %2459 : tensor<256xf64>
        %2483 = stablehlo.slice %1614 [0:256, 11:12, 9:10] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2484 = stablehlo.reshape %2483 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2485 = stablehlo.multiply %1680, %1670 : tensor<256xf64>
        %cst_297 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2486 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2487 = stablehlo.add %2486, %2485 : tensor<256xf64>
        %2488 = stablehlo.multiply %1757, %1743 : tensor<256xf64>
        %2489 = stablehlo.add %2487, %2488 : tensor<256xf64>
        %2490 = stablehlo.multiply %1847, %1829 : tensor<256xf64>
        %2491 = stablehlo.add %2489, %2490 : tensor<256xf64>
        %2492 = stablehlo.multiply %1946, %1924 : tensor<256xf64>
        %2493 = stablehlo.add %2491, %2492 : tensor<256xf64>
        %2494 = stablehlo.multiply %2050, %2024 : tensor<256xf64>
        %2495 = stablehlo.add %2493, %2494 : tensor<256xf64>
        %2496 = stablehlo.multiply %2155, %2125 : tensor<256xf64>
        %2497 = stablehlo.add %2495, %2496 : tensor<256xf64>
        %2498 = stablehlo.multiply %2257, %2223 : tensor<256xf64>
        %2499 = stablehlo.add %2497, %2498 : tensor<256xf64>
        %2500 = stablehlo.multiply %2352, %2314 : tensor<256xf64>
        %2501 = stablehlo.add %2499, %2500 : tensor<256xf64>
        %2502 = stablehlo.multiply %2436, %2394 : tensor<256xf64>
        %2503 = stablehlo.add %2501, %2502 : tensor<256xf64>
        %2504 = stablehlo.subtract %2484, %2503 : tensor<256xf64>
        %2505 = stablehlo.divide %2504, %2459 : tensor<256xf64>
        %2506 = stablehlo.slice %1614 [0:256, 10:11, 10:11] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2507 = stablehlo.reshape %2506 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2508 = stablehlo.multiply %1675, %1675 : tensor<256xf64>
        %cst_298 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2509 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2510 = stablehlo.add %2509, %2508 : tensor<256xf64>
        %2511 = stablehlo.multiply %1750, %1750 : tensor<256xf64>
        %2512 = stablehlo.add %2510, %2511 : tensor<256xf64>
        %2513 = stablehlo.multiply %1838, %1838 : tensor<256xf64>
        %2514 = stablehlo.add %2512, %2513 : tensor<256xf64>
        %2515 = stablehlo.multiply %1935, %1935 : tensor<256xf64>
        %2516 = stablehlo.add %2514, %2515 : tensor<256xf64>
        %2517 = stablehlo.multiply %2037, %2037 : tensor<256xf64>
        %2518 = stablehlo.add %2516, %2517 : tensor<256xf64>
        %2519 = stablehlo.multiply %2140, %2140 : tensor<256xf64>
        %2520 = stablehlo.add %2518, %2519 : tensor<256xf64>
        %2521 = stablehlo.multiply %2240, %2240 : tensor<256xf64>
        %2522 = stablehlo.add %2520, %2521 : tensor<256xf64>
        %2523 = stablehlo.multiply %2333, %2333 : tensor<256xf64>
        %2524 = stablehlo.add %2522, %2523 : tensor<256xf64>
        %2525 = stablehlo.multiply %2415, %2415 : tensor<256xf64>
        %2526 = stablehlo.add %2524, %2525 : tensor<256xf64>
        %2527 = stablehlo.multiply %2482, %2482 : tensor<256xf64>
        %2528 = stablehlo.add %2526, %2527 : tensor<256xf64>
        %2529 = stablehlo.subtract %2507, %2528 : tensor<256xf64>
        %2530 = stablehlo.sqrt %2529 : tensor<256xf64>
        %2531 = stablehlo.slice %1614 [0:256, 11:12, 10:11] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2532 = stablehlo.reshape %2531 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2533 = stablehlo.multiply %1680, %1675 : tensor<256xf64>
        %cst_299 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2534 = stablehlo.broadcast_in_dim %cst_299, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2535 = stablehlo.add %2534, %2533 : tensor<256xf64>
        %2536 = stablehlo.multiply %1757, %1750 : tensor<256xf64>
        %2537 = stablehlo.add %2535, %2536 : tensor<256xf64>
        %2538 = stablehlo.multiply %1847, %1838 : tensor<256xf64>
        %2539 = stablehlo.add %2537, %2538 : tensor<256xf64>
        %2540 = stablehlo.multiply %1946, %1935 : tensor<256xf64>
        %2541 = stablehlo.add %2539, %2540 : tensor<256xf64>
        %2542 = stablehlo.multiply %2050, %2037 : tensor<256xf64>
        %2543 = stablehlo.add %2541, %2542 : tensor<256xf64>
        %2544 = stablehlo.multiply %2155, %2140 : tensor<256xf64>
        %2545 = stablehlo.add %2543, %2544 : tensor<256xf64>
        %2546 = stablehlo.multiply %2257, %2240 : tensor<256xf64>
        %2547 = stablehlo.add %2545, %2546 : tensor<256xf64>
        %2548 = stablehlo.multiply %2352, %2333 : tensor<256xf64>
        %2549 = stablehlo.add %2547, %2548 : tensor<256xf64>
        %2550 = stablehlo.multiply %2436, %2415 : tensor<256xf64>
        %2551 = stablehlo.add %2549, %2550 : tensor<256xf64>
        %2552 = stablehlo.multiply %2505, %2482 : tensor<256xf64>
        %2553 = stablehlo.add %2551, %2552 : tensor<256xf64>
        %2554 = stablehlo.subtract %2532, %2553 : tensor<256xf64>
        %2555 = stablehlo.divide %2554, %2530 : tensor<256xf64>
        %2556 = stablehlo.slice %1614 [0:256, 11:12, 11:12] : (tensor<256x12x12xf64>) -> tensor<256x1x1xf64>
        %2557 = stablehlo.reshape %2556 : (tensor<256x1x1xf64>) -> tensor<256xf64>
        %2558 = stablehlo.multiply %1680, %1680 : tensor<256xf64>
        %cst_300 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2559 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2560 = stablehlo.add %2559, %2558 : tensor<256xf64>
        %2561 = stablehlo.multiply %1757, %1757 : tensor<256xf64>
        %2562 = stablehlo.add %2560, %2561 : tensor<256xf64>
        %2563 = stablehlo.multiply %1847, %1847 : tensor<256xf64>
        %2564 = stablehlo.add %2562, %2563 : tensor<256xf64>
        %2565 = stablehlo.multiply %1946, %1946 : tensor<256xf64>
        %2566 = stablehlo.add %2564, %2565 : tensor<256xf64>
        %2567 = stablehlo.multiply %2050, %2050 : tensor<256xf64>
        %2568 = stablehlo.add %2566, %2567 : tensor<256xf64>
        %2569 = stablehlo.multiply %2155, %2155 : tensor<256xf64>
        %2570 = stablehlo.add %2568, %2569 : tensor<256xf64>
        %2571 = stablehlo.multiply %2257, %2257 : tensor<256xf64>
        %2572 = stablehlo.add %2570, %2571 : tensor<256xf64>
        %2573 = stablehlo.multiply %2352, %2352 : tensor<256xf64>
        %2574 = stablehlo.add %2572, %2573 : tensor<256xf64>
        %2575 = stablehlo.multiply %2436, %2436 : tensor<256xf64>
        %2576 = stablehlo.add %2574, %2575 : tensor<256xf64>
        %2577 = stablehlo.multiply %2505, %2505 : tensor<256xf64>
        %2578 = stablehlo.add %2576, %2577 : tensor<256xf64>
        %2579 = stablehlo.multiply %2555, %2555 : tensor<256xf64>
        %2580 = stablehlo.add %2578, %2579 : tensor<256xf64>
        %2581 = stablehlo.subtract %2557, %2580 : tensor<256xf64>
        %2582 = stablehlo.sqrt %2581 : tensor<256xf64>
        %2583 = stablehlo.slice %1620 [0:256, 0:1] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2584 = stablehlo.reshape %2583 : (tensor<256x1xf64>) -> tensor<256xf64>
        %cst_301 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2585 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2586 = stablehlo.subtract %2584, %2585 : tensor<256xf64>
        %2587 = stablehlo.divide %2586, %1625 : tensor<256xf64>
        %2588 = stablehlo.slice %1620 [0:256, 1:2] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2589 = stablehlo.reshape %2588 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2590 = stablehlo.multiply %1630, %2587 : tensor<256xf64>
        %cst_302 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2591 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2592 = stablehlo.add %2591, %2590 : tensor<256xf64>
        %2593 = stablehlo.subtract %2589, %2592 : tensor<256xf64>
        %2594 = stablehlo.divide %2593, %1687 : tensor<256xf64>
        %2595 = stablehlo.slice %1620 [0:256, 2:3] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2596 = stablehlo.reshape %2595 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2597 = stablehlo.multiply %1635, %2587 : tensor<256xf64>
        %cst_303 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2598 = stablehlo.broadcast_in_dim %cst_303, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2599 = stablehlo.add %2598, %2597 : tensor<256xf64>
        %2600 = stablehlo.multiply %1694, %2594 : tensor<256xf64>
        %2601 = stablehlo.add %2599, %2600 : tensor<256xf64>
        %2602 = stablehlo.subtract %2596, %2601 : tensor<256xf64>
        %2603 = stablehlo.divide %2602, %1766 : tensor<256xf64>
        %2604 = stablehlo.slice %1620 [0:256, 3:4] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2605 = stablehlo.reshape %2604 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2606 = stablehlo.multiply %1640, %2587 : tensor<256xf64>
        %cst_304 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2607 = stablehlo.broadcast_in_dim %cst_304, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2608 = stablehlo.add %2607, %2606 : tensor<256xf64>
        %2609 = stablehlo.multiply %1701, %2594 : tensor<256xf64>
        %2610 = stablehlo.add %2608, %2609 : tensor<256xf64>
        %2611 = stablehlo.multiply %1775, %2603 : tensor<256xf64>
        %2612 = stablehlo.add %2610, %2611 : tensor<256xf64>
        %2613 = stablehlo.subtract %2605, %2612 : tensor<256xf64>
        %2614 = stablehlo.divide %2613, %1858 : tensor<256xf64>
        %2615 = stablehlo.slice %1620 [0:256, 4:5] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2616 = stablehlo.reshape %2615 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2617 = stablehlo.multiply %1645, %2587 : tensor<256xf64>
        %cst_305 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2618 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2619 = stablehlo.add %2618, %2617 : tensor<256xf64>
        %2620 = stablehlo.multiply %1708, %2594 : tensor<256xf64>
        %2621 = stablehlo.add %2619, %2620 : tensor<256xf64>
        %2622 = stablehlo.multiply %1784, %2603 : tensor<256xf64>
        %2623 = stablehlo.add %2621, %2622 : tensor<256xf64>
        %2624 = stablehlo.multiply %1869, %2614 : tensor<256xf64>
        %2625 = stablehlo.add %2623, %2624 : tensor<256xf64>
        %2626 = stablehlo.subtract %2616, %2625 : tensor<256xf64>
        %2627 = stablehlo.divide %2626, %1959 : tensor<256xf64>
        %2628 = stablehlo.slice %1620 [0:256, 5:6] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2629 = stablehlo.reshape %2628 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2630 = stablehlo.multiply %1650, %2587 : tensor<256xf64>
        %cst_306 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2631 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2632 = stablehlo.add %2631, %2630 : tensor<256xf64>
        %2633 = stablehlo.multiply %1715, %2594 : tensor<256xf64>
        %2634 = stablehlo.add %2632, %2633 : tensor<256xf64>
        %2635 = stablehlo.multiply %1793, %2603 : tensor<256xf64>
        %2636 = stablehlo.add %2634, %2635 : tensor<256xf64>
        %2637 = stablehlo.multiply %1880, %2614 : tensor<256xf64>
        %2638 = stablehlo.add %2636, %2637 : tensor<256xf64>
        %2639 = stablehlo.multiply %1972, %2627 : tensor<256xf64>
        %2640 = stablehlo.add %2638, %2639 : tensor<256xf64>
        %2641 = stablehlo.subtract %2629, %2640 : tensor<256xf64>
        %2642 = stablehlo.divide %2641, %2065 : tensor<256xf64>
        %2643 = stablehlo.slice %1620 [0:256, 6:7] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2644 = stablehlo.reshape %2643 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2645 = stablehlo.multiply %1655, %2587 : tensor<256xf64>
        %cst_307 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2646 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2647 = stablehlo.add %2646, %2645 : tensor<256xf64>
        %2648 = stablehlo.multiply %1722, %2594 : tensor<256xf64>
        %2649 = stablehlo.add %2647, %2648 : tensor<256xf64>
        %2650 = stablehlo.multiply %1802, %2603 : tensor<256xf64>
        %2651 = stablehlo.add %2649, %2650 : tensor<256xf64>
        %2652 = stablehlo.multiply %1891, %2614 : tensor<256xf64>
        %2653 = stablehlo.add %2651, %2652 : tensor<256xf64>
        %2654 = stablehlo.multiply %1985, %2627 : tensor<256xf64>
        %2655 = stablehlo.add %2653, %2654 : tensor<256xf64>
        %2656 = stablehlo.multiply %2080, %2642 : tensor<256xf64>
        %2657 = stablehlo.add %2655, %2656 : tensor<256xf64>
        %2658 = stablehlo.subtract %2644, %2657 : tensor<256xf64>
        %2659 = stablehlo.divide %2658, %2172 : tensor<256xf64>
        %2660 = stablehlo.slice %1620 [0:256, 7:8] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2661 = stablehlo.reshape %2660 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2662 = stablehlo.multiply %1660, %2587 : tensor<256xf64>
        %cst_308 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2663 = stablehlo.broadcast_in_dim %cst_308, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2664 = stablehlo.add %2663, %2662 : tensor<256xf64>
        %2665 = stablehlo.multiply %1729, %2594 : tensor<256xf64>
        %2666 = stablehlo.add %2664, %2665 : tensor<256xf64>
        %2667 = stablehlo.multiply %1811, %2603 : tensor<256xf64>
        %2668 = stablehlo.add %2666, %2667 : tensor<256xf64>
        %2669 = stablehlo.multiply %1902, %2614 : tensor<256xf64>
        %2670 = stablehlo.add %2668, %2669 : tensor<256xf64>
        %2671 = stablehlo.multiply %1998, %2627 : tensor<256xf64>
        %2672 = stablehlo.add %2670, %2671 : tensor<256xf64>
        %2673 = stablehlo.multiply %2095, %2642 : tensor<256xf64>
        %2674 = stablehlo.add %2672, %2673 : tensor<256xf64>
        %2675 = stablehlo.multiply %2189, %2659 : tensor<256xf64>
        %2676 = stablehlo.add %2674, %2675 : tensor<256xf64>
        %2677 = stablehlo.subtract %2661, %2676 : tensor<256xf64>
        %2678 = stablehlo.divide %2677, %2276 : tensor<256xf64>
        %2679 = stablehlo.slice %1620 [0:256, 8:9] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2680 = stablehlo.reshape %2679 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2681 = stablehlo.multiply %1665, %2587 : tensor<256xf64>
        %cst_309 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2682 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2683 = stablehlo.add %2682, %2681 : tensor<256xf64>
        %2684 = stablehlo.multiply %1736, %2594 : tensor<256xf64>
        %2685 = stablehlo.add %2683, %2684 : tensor<256xf64>
        %2686 = stablehlo.multiply %1820, %2603 : tensor<256xf64>
        %2687 = stablehlo.add %2685, %2686 : tensor<256xf64>
        %2688 = stablehlo.multiply %1913, %2614 : tensor<256xf64>
        %2689 = stablehlo.add %2687, %2688 : tensor<256xf64>
        %2690 = stablehlo.multiply %2011, %2627 : tensor<256xf64>
        %2691 = stablehlo.add %2689, %2690 : tensor<256xf64>
        %2692 = stablehlo.multiply %2110, %2642 : tensor<256xf64>
        %2693 = stablehlo.add %2691, %2692 : tensor<256xf64>
        %2694 = stablehlo.multiply %2206, %2659 : tensor<256xf64>
        %2695 = stablehlo.add %2693, %2694 : tensor<256xf64>
        %2696 = stablehlo.multiply %2295, %2678 : tensor<256xf64>
        %2697 = stablehlo.add %2695, %2696 : tensor<256xf64>
        %2698 = stablehlo.subtract %2680, %2697 : tensor<256xf64>
        %2699 = stablehlo.divide %2698, %2373 : tensor<256xf64>
        %2700 = stablehlo.slice %1620 [0:256, 9:10] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2701 = stablehlo.reshape %2700 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2702 = stablehlo.multiply %1670, %2587 : tensor<256xf64>
        %cst_310 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2703 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2704 = stablehlo.add %2703, %2702 : tensor<256xf64>
        %2705 = stablehlo.multiply %1743, %2594 : tensor<256xf64>
        %2706 = stablehlo.add %2704, %2705 : tensor<256xf64>
        %2707 = stablehlo.multiply %1829, %2603 : tensor<256xf64>
        %2708 = stablehlo.add %2706, %2707 : tensor<256xf64>
        %2709 = stablehlo.multiply %1924, %2614 : tensor<256xf64>
        %2710 = stablehlo.add %2708, %2709 : tensor<256xf64>
        %2711 = stablehlo.multiply %2024, %2627 : tensor<256xf64>
        %2712 = stablehlo.add %2710, %2711 : tensor<256xf64>
        %2713 = stablehlo.multiply %2125, %2642 : tensor<256xf64>
        %2714 = stablehlo.add %2712, %2713 : tensor<256xf64>
        %2715 = stablehlo.multiply %2223, %2659 : tensor<256xf64>
        %2716 = stablehlo.add %2714, %2715 : tensor<256xf64>
        %2717 = stablehlo.multiply %2314, %2678 : tensor<256xf64>
        %2718 = stablehlo.add %2716, %2717 : tensor<256xf64>
        %2719 = stablehlo.multiply %2394, %2699 : tensor<256xf64>
        %2720 = stablehlo.add %2718, %2719 : tensor<256xf64>
        %2721 = stablehlo.subtract %2701, %2720 : tensor<256xf64>
        %2722 = stablehlo.divide %2721, %2459 : tensor<256xf64>
        %2723 = stablehlo.slice %1620 [0:256, 10:11] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2724 = stablehlo.reshape %2723 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2725 = stablehlo.multiply %1675, %2587 : tensor<256xf64>
        %cst_311 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2726 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2727 = stablehlo.add %2726, %2725 : tensor<256xf64>
        %2728 = stablehlo.multiply %1750, %2594 : tensor<256xf64>
        %2729 = stablehlo.add %2727, %2728 : tensor<256xf64>
        %2730 = stablehlo.multiply %1838, %2603 : tensor<256xf64>
        %2731 = stablehlo.add %2729, %2730 : tensor<256xf64>
        %2732 = stablehlo.multiply %1935, %2614 : tensor<256xf64>
        %2733 = stablehlo.add %2731, %2732 : tensor<256xf64>
        %2734 = stablehlo.multiply %2037, %2627 : tensor<256xf64>
        %2735 = stablehlo.add %2733, %2734 : tensor<256xf64>
        %2736 = stablehlo.multiply %2140, %2642 : tensor<256xf64>
        %2737 = stablehlo.add %2735, %2736 : tensor<256xf64>
        %2738 = stablehlo.multiply %2240, %2659 : tensor<256xf64>
        %2739 = stablehlo.add %2737, %2738 : tensor<256xf64>
        %2740 = stablehlo.multiply %2333, %2678 : tensor<256xf64>
        %2741 = stablehlo.add %2739, %2740 : tensor<256xf64>
        %2742 = stablehlo.multiply %2415, %2699 : tensor<256xf64>
        %2743 = stablehlo.add %2741, %2742 : tensor<256xf64>
        %2744 = stablehlo.multiply %2482, %2722 : tensor<256xf64>
        %2745 = stablehlo.add %2743, %2744 : tensor<256xf64>
        %2746 = stablehlo.subtract %2724, %2745 : tensor<256xf64>
        %2747 = stablehlo.divide %2746, %2530 : tensor<256xf64>
        %2748 = stablehlo.slice %1620 [0:256, 11:12] : (tensor<256x12xf64>) -> tensor<256x1xf64>
        %2749 = stablehlo.reshape %2748 : (tensor<256x1xf64>) -> tensor<256xf64>
        %2750 = stablehlo.multiply %1680, %2587 : tensor<256xf64>
        %cst_312 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2751 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2752 = stablehlo.add %2751, %2750 : tensor<256xf64>
        %2753 = stablehlo.multiply %1757, %2594 : tensor<256xf64>
        %2754 = stablehlo.add %2752, %2753 : tensor<256xf64>
        %2755 = stablehlo.multiply %1847, %2603 : tensor<256xf64>
        %2756 = stablehlo.add %2754, %2755 : tensor<256xf64>
        %2757 = stablehlo.multiply %1946, %2614 : tensor<256xf64>
        %2758 = stablehlo.add %2756, %2757 : tensor<256xf64>
        %2759 = stablehlo.multiply %2050, %2627 : tensor<256xf64>
        %2760 = stablehlo.add %2758, %2759 : tensor<256xf64>
        %2761 = stablehlo.multiply %2155, %2642 : tensor<256xf64>
        %2762 = stablehlo.add %2760, %2761 : tensor<256xf64>
        %2763 = stablehlo.multiply %2257, %2659 : tensor<256xf64>
        %2764 = stablehlo.add %2762, %2763 : tensor<256xf64>
        %2765 = stablehlo.multiply %2352, %2678 : tensor<256xf64>
        %2766 = stablehlo.add %2764, %2765 : tensor<256xf64>
        %2767 = stablehlo.multiply %2436, %2699 : tensor<256xf64>
        %2768 = stablehlo.add %2766, %2767 : tensor<256xf64>
        %2769 = stablehlo.multiply %2505, %2722 : tensor<256xf64>
        %2770 = stablehlo.add %2768, %2769 : tensor<256xf64>
        %2771 = stablehlo.multiply %2555, %2747 : tensor<256xf64>
        %2772 = stablehlo.add %2770, %2771 : tensor<256xf64>
        %2773 = stablehlo.subtract %2749, %2772 : tensor<256xf64>
        %2774 = stablehlo.divide %2773, %2582 : tensor<256xf64>
        %cst_313 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2775 = stablehlo.broadcast_in_dim %cst_313, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2776 = stablehlo.subtract %2774, %2775 : tensor<256xf64>
        %2777 = stablehlo.divide %2776, %2582 : tensor<256xf64>
        %2778 = stablehlo.multiply %2555, %2777 : tensor<256xf64>
        %cst_314 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2779 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2780 = stablehlo.add %2779, %2778 : tensor<256xf64>
        %2781 = stablehlo.subtract %2747, %2780 : tensor<256xf64>
        %2782 = stablehlo.divide %2781, %2530 : tensor<256xf64>
        %2783 = stablehlo.multiply %2482, %2782 : tensor<256xf64>
        %cst_315 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2784 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2785 = stablehlo.add %2784, %2783 : tensor<256xf64>
        %2786 = stablehlo.multiply %2505, %2777 : tensor<256xf64>
        %2787 = stablehlo.add %2785, %2786 : tensor<256xf64>
        %2788 = stablehlo.subtract %2722, %2787 : tensor<256xf64>
        %2789 = stablehlo.divide %2788, %2459 : tensor<256xf64>
        %2790 = stablehlo.multiply %2394, %2789 : tensor<256xf64>
        %cst_316 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2791 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2792 = stablehlo.add %2791, %2790 : tensor<256xf64>
        %2793 = stablehlo.multiply %2415, %2782 : tensor<256xf64>
        %2794 = stablehlo.add %2792, %2793 : tensor<256xf64>
        %2795 = stablehlo.multiply %2436, %2777 : tensor<256xf64>
        %2796 = stablehlo.add %2794, %2795 : tensor<256xf64>
        %2797 = stablehlo.subtract %2699, %2796 : tensor<256xf64>
        %2798 = stablehlo.divide %2797, %2373 : tensor<256xf64>
        %2799 = stablehlo.multiply %2295, %2798 : tensor<256xf64>
        %cst_317 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2800 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2801 = stablehlo.add %2800, %2799 : tensor<256xf64>
        %2802 = stablehlo.multiply %2314, %2789 : tensor<256xf64>
        %2803 = stablehlo.add %2801, %2802 : tensor<256xf64>
        %2804 = stablehlo.multiply %2333, %2782 : tensor<256xf64>
        %2805 = stablehlo.add %2803, %2804 : tensor<256xf64>
        %2806 = stablehlo.multiply %2352, %2777 : tensor<256xf64>
        %2807 = stablehlo.add %2805, %2806 : tensor<256xf64>
        %2808 = stablehlo.subtract %2678, %2807 : tensor<256xf64>
        %2809 = stablehlo.divide %2808, %2276 : tensor<256xf64>
        %2810 = stablehlo.multiply %2189, %2809 : tensor<256xf64>
        %cst_318 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2811 = stablehlo.broadcast_in_dim %cst_318, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2812 = stablehlo.add %2811, %2810 : tensor<256xf64>
        %2813 = stablehlo.multiply %2206, %2798 : tensor<256xf64>
        %2814 = stablehlo.add %2812, %2813 : tensor<256xf64>
        %2815 = stablehlo.multiply %2223, %2789 : tensor<256xf64>
        %2816 = stablehlo.add %2814, %2815 : tensor<256xf64>
        %2817 = stablehlo.multiply %2240, %2782 : tensor<256xf64>
        %2818 = stablehlo.add %2816, %2817 : tensor<256xf64>
        %2819 = stablehlo.multiply %2257, %2777 : tensor<256xf64>
        %2820 = stablehlo.add %2818, %2819 : tensor<256xf64>
        %2821 = stablehlo.subtract %2659, %2820 : tensor<256xf64>
        %2822 = stablehlo.divide %2821, %2172 : tensor<256xf64>
        %2823 = stablehlo.multiply %2080, %2822 : tensor<256xf64>
        %cst_319 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2824 = stablehlo.broadcast_in_dim %cst_319, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2825 = stablehlo.add %2824, %2823 : tensor<256xf64>
        %2826 = stablehlo.multiply %2095, %2809 : tensor<256xf64>
        %2827 = stablehlo.add %2825, %2826 : tensor<256xf64>
        %2828 = stablehlo.multiply %2110, %2798 : tensor<256xf64>
        %2829 = stablehlo.add %2827, %2828 : tensor<256xf64>
        %2830 = stablehlo.multiply %2125, %2789 : tensor<256xf64>
        %2831 = stablehlo.add %2829, %2830 : tensor<256xf64>
        %2832 = stablehlo.multiply %2140, %2782 : tensor<256xf64>
        %2833 = stablehlo.add %2831, %2832 : tensor<256xf64>
        %2834 = stablehlo.multiply %2155, %2777 : tensor<256xf64>
        %2835 = stablehlo.add %2833, %2834 : tensor<256xf64>
        %2836 = stablehlo.subtract %2642, %2835 : tensor<256xf64>
        %2837 = stablehlo.divide %2836, %2065 : tensor<256xf64>
        %2838 = stablehlo.multiply %1972, %2837 : tensor<256xf64>
        %cst_320 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2839 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2840 = stablehlo.add %2839, %2838 : tensor<256xf64>
        %2841 = stablehlo.multiply %1985, %2822 : tensor<256xf64>
        %2842 = stablehlo.add %2840, %2841 : tensor<256xf64>
        %2843 = stablehlo.multiply %1998, %2809 : tensor<256xf64>
        %2844 = stablehlo.add %2842, %2843 : tensor<256xf64>
        %2845 = stablehlo.multiply %2011, %2798 : tensor<256xf64>
        %2846 = stablehlo.add %2844, %2845 : tensor<256xf64>
        %2847 = stablehlo.multiply %2024, %2789 : tensor<256xf64>
        %2848 = stablehlo.add %2846, %2847 : tensor<256xf64>
        %2849 = stablehlo.multiply %2037, %2782 : tensor<256xf64>
        %2850 = stablehlo.add %2848, %2849 : tensor<256xf64>
        %2851 = stablehlo.multiply %2050, %2777 : tensor<256xf64>
        %2852 = stablehlo.add %2850, %2851 : tensor<256xf64>
        %2853 = stablehlo.subtract %2627, %2852 : tensor<256xf64>
        %2854 = stablehlo.divide %2853, %1959 : tensor<256xf64>
        %2855 = stablehlo.multiply %1869, %2854 : tensor<256xf64>
        %cst_321 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2856 = stablehlo.broadcast_in_dim %cst_321, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2857 = stablehlo.add %2856, %2855 : tensor<256xf64>
        %2858 = stablehlo.multiply %1880, %2837 : tensor<256xf64>
        %2859 = stablehlo.add %2857, %2858 : tensor<256xf64>
        %2860 = stablehlo.multiply %1891, %2822 : tensor<256xf64>
        %2861 = stablehlo.add %2859, %2860 : tensor<256xf64>
        %2862 = stablehlo.multiply %1902, %2809 : tensor<256xf64>
        %2863 = stablehlo.add %2861, %2862 : tensor<256xf64>
        %2864 = stablehlo.multiply %1913, %2798 : tensor<256xf64>
        %2865 = stablehlo.add %2863, %2864 : tensor<256xf64>
        %2866 = stablehlo.multiply %1924, %2789 : tensor<256xf64>
        %2867 = stablehlo.add %2865, %2866 : tensor<256xf64>
        %2868 = stablehlo.multiply %1935, %2782 : tensor<256xf64>
        %2869 = stablehlo.add %2867, %2868 : tensor<256xf64>
        %2870 = stablehlo.multiply %1946, %2777 : tensor<256xf64>
        %2871 = stablehlo.add %2869, %2870 : tensor<256xf64>
        %2872 = stablehlo.subtract %2614, %2871 : tensor<256xf64>
        %2873 = stablehlo.divide %2872, %1858 : tensor<256xf64>
        %2874 = stablehlo.multiply %1775, %2873 : tensor<256xf64>
        %cst_322 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2875 = stablehlo.broadcast_in_dim %cst_322, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2876 = stablehlo.add %2875, %2874 : tensor<256xf64>
        %2877 = stablehlo.multiply %1784, %2854 : tensor<256xf64>
        %2878 = stablehlo.add %2876, %2877 : tensor<256xf64>
        %2879 = stablehlo.multiply %1793, %2837 : tensor<256xf64>
        %2880 = stablehlo.add %2878, %2879 : tensor<256xf64>
        %2881 = stablehlo.multiply %1802, %2822 : tensor<256xf64>
        %2882 = stablehlo.add %2880, %2881 : tensor<256xf64>
        %2883 = stablehlo.multiply %1811, %2809 : tensor<256xf64>
        %2884 = stablehlo.add %2882, %2883 : tensor<256xf64>
        %2885 = stablehlo.multiply %1820, %2798 : tensor<256xf64>
        %2886 = stablehlo.add %2884, %2885 : tensor<256xf64>
        %2887 = stablehlo.multiply %1829, %2789 : tensor<256xf64>
        %2888 = stablehlo.add %2886, %2887 : tensor<256xf64>
        %2889 = stablehlo.multiply %1838, %2782 : tensor<256xf64>
        %2890 = stablehlo.add %2888, %2889 : tensor<256xf64>
        %2891 = stablehlo.multiply %1847, %2777 : tensor<256xf64>
        %2892 = stablehlo.add %2890, %2891 : tensor<256xf64>
        %2893 = stablehlo.subtract %2603, %2892 : tensor<256xf64>
        %2894 = stablehlo.divide %2893, %1766 : tensor<256xf64>
        %2895 = stablehlo.multiply %1694, %2894 : tensor<256xf64>
        %cst_323 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2896 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2897 = stablehlo.add %2896, %2895 : tensor<256xf64>
        %2898 = stablehlo.multiply %1701, %2873 : tensor<256xf64>
        %2899 = stablehlo.add %2897, %2898 : tensor<256xf64>
        %2900 = stablehlo.multiply %1708, %2854 : tensor<256xf64>
        %2901 = stablehlo.add %2899, %2900 : tensor<256xf64>
        %2902 = stablehlo.multiply %1715, %2837 : tensor<256xf64>
        %2903 = stablehlo.add %2901, %2902 : tensor<256xf64>
        %2904 = stablehlo.multiply %1722, %2822 : tensor<256xf64>
        %2905 = stablehlo.add %2903, %2904 : tensor<256xf64>
        %2906 = stablehlo.multiply %1729, %2809 : tensor<256xf64>
        %2907 = stablehlo.add %2905, %2906 : tensor<256xf64>
        %2908 = stablehlo.multiply %1736, %2798 : tensor<256xf64>
        %2909 = stablehlo.add %2907, %2908 : tensor<256xf64>
        %2910 = stablehlo.multiply %1743, %2789 : tensor<256xf64>
        %2911 = stablehlo.add %2909, %2910 : tensor<256xf64>
        %2912 = stablehlo.multiply %1750, %2782 : tensor<256xf64>
        %2913 = stablehlo.add %2911, %2912 : tensor<256xf64>
        %2914 = stablehlo.multiply %1757, %2777 : tensor<256xf64>
        %2915 = stablehlo.add %2913, %2914 : tensor<256xf64>
        %2916 = stablehlo.subtract %2594, %2915 : tensor<256xf64>
        %2917 = stablehlo.divide %2916, %1687 : tensor<256xf64>
        %2918 = stablehlo.multiply %1630, %2917 : tensor<256xf64>
        %cst_324 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2919 = stablehlo.broadcast_in_dim %cst_324, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2920 = stablehlo.add %2919, %2918 : tensor<256xf64>
        %2921 = stablehlo.multiply %1635, %2894 : tensor<256xf64>
        %2922 = stablehlo.add %2920, %2921 : tensor<256xf64>
        %2923 = stablehlo.multiply %1640, %2873 : tensor<256xf64>
        %2924 = stablehlo.add %2922, %2923 : tensor<256xf64>
        %2925 = stablehlo.multiply %1645, %2854 : tensor<256xf64>
        %2926 = stablehlo.add %2924, %2925 : tensor<256xf64>
        %2927 = stablehlo.multiply %1650, %2837 : tensor<256xf64>
        %2928 = stablehlo.add %2926, %2927 : tensor<256xf64>
        %2929 = stablehlo.multiply %1655, %2822 : tensor<256xf64>
        %2930 = stablehlo.add %2928, %2929 : tensor<256xf64>
        %2931 = stablehlo.multiply %1660, %2809 : tensor<256xf64>
        %2932 = stablehlo.add %2930, %2931 : tensor<256xf64>
        %2933 = stablehlo.multiply %1665, %2798 : tensor<256xf64>
        %2934 = stablehlo.add %2932, %2933 : tensor<256xf64>
        %2935 = stablehlo.multiply %1670, %2789 : tensor<256xf64>
        %2936 = stablehlo.add %2934, %2935 : tensor<256xf64>
        %2937 = stablehlo.multiply %1675, %2782 : tensor<256xf64>
        %2938 = stablehlo.add %2936, %2937 : tensor<256xf64>
        %2939 = stablehlo.multiply %1680, %2777 : tensor<256xf64>
        %2940 = stablehlo.add %2938, %2939 : tensor<256xf64>
        %2941 = stablehlo.subtract %2587, %2940 : tensor<256xf64>
        %2942 = stablehlo.divide %2941, %1625 : tensor<256xf64>
        %2943 = stablehlo.broadcast_in_dim %2942, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2944 = stablehlo.broadcast_in_dim %2917, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2945 = stablehlo.broadcast_in_dim %2894, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2946 = stablehlo.broadcast_in_dim %2873, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2947 = stablehlo.broadcast_in_dim %2854, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2948 = stablehlo.broadcast_in_dim %2837, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2949 = stablehlo.broadcast_in_dim %2822, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2950 = stablehlo.broadcast_in_dim %2809, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2951 = stablehlo.broadcast_in_dim %2798, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2952 = stablehlo.broadcast_in_dim %2789, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2953 = stablehlo.broadcast_in_dim %2782, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2954 = stablehlo.broadcast_in_dim %2777, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
        %2955 = stablehlo.concatenate %2943, %2944, %2945, %2946, %2947, %2948, %2949, %2950, %2951, %2952, %2953, %2954, dim = 0 : (tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>, tensor<1x256xf64>) -> tensor<12x256xf64>
        %2956 = stablehlo.transpose %2955, dims = [1, 0] : (tensor<12x256xf64>) -> tensor<256x12xf64>
        %2957 = stablehlo.add %iterArg_199, %2956 : tensor<256x12xf64>
        %2958 = stablehlo.dot_general %iterArg_185, %2957, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<192x12xf64>, tensor<256x12xf64>) -> tensor<192x256xf64>
        %2959 = stablehlo.transpose %2958, dims = [1, 0] : (tensor<192x256xf64>) -> tensor<256x192xf64>
        %2960 = stablehlo.add %2959, %iterArg_186 : tensor<256x192xf64>
        %2961 = stablehlo.exponential %2960 : tensor<256x192xf64>
        %cst_325 = stablehlo.constant dense<1.000000e-300> : tensor<f64>
        %2962 = stablehlo.broadcast_in_dim %cst_325, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
        %2963 = stablehlo.maximum %2961, %2962 : tensor<256x192xf64>
        %cst_326 = stablehlo.constant dense<-690.77552789821368> : tensor<f64>
        %cst_327 = stablehlo.constant dense<690.77552789821368> : tensor<f64>
        %2964 = func.call @clip(%2960, %cst_326, %cst_327) : (tensor<256x192xf64>, tensor<f64>, tensor<f64>) -> tensor<256x192xf64>
        %cst_328 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2965 = stablehlo.broadcast_in_dim %cst_328, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2966 = stablehlo.compare GT, %iterArg_187, %2965, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
        %cst_329 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %2967 = func.call @_where_3(%2966, %iterArg_187, %cst_329) : (tensor<256xi1>, tensor<256xf64>, tensor<f64>) -> tensor<256xf64>
        %cst_330 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %2968 = stablehlo.broadcast_in_dim %cst_330, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2969 = stablehlo.divide %2968, %2967 : tensor<256xf64>
        %2970 = stablehlo.broadcast_in_dim %2969, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
        %2971 = stablehlo.broadcast_in_dim %2970, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
        %2972 = stablehlo.add %iterArg_188, %2971 : tensor<256x192xf64>
        %2973 = stablehlo.broadcast_in_dim %iterArg_187, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
        %2974 = stablehlo.broadcast_in_dim %2973, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
        %2975 = stablehlo.multiply %2974, %2963 : tensor<256x192xf64>
        %2976 = stablehlo.log_plus_one %2975 : tensor<256x192xf64>
        %2977 = stablehlo.multiply %2972, %2976 : tensor<256x192xf64>
        %2978 = func.call @_where_5(%2966, %2977, %2963) : (tensor<256xi1>, tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xf64>
        %2979 = stablehlo.multiply %iterArg_188, %2964 : tensor<256x192xf64>
        %2980 = stablehlo.subtract %2978, %2979 : tensor<256x192xf64>
        %2981 = stablehlo.multiply %iterArg_189, %2980 : tensor<256x192xf64>
        %cst_331 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
        %2982 = stablehlo.reduce(%2981 init: %cst_331) applies stablehlo.add across dimensions = [1] : (tensor<256x192xf64>, tensor<f64>) -> tensor<256xf64>
        %2983 = stablehlo.add %iterArg_198, %2982 : tensor<256xf64>
        %cst_332 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
        %2984 = stablehlo.broadcast_in_dim %cst_332, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2985 = stablehlo.multiply %2984, %2983 : tensor<256xf64>
        %c_333 = stablehlo.constant dense<0> : tensor<i64>
        %2986 = stablehlo.broadcast_in_dim %c_333, dims = [] : (tensor<i64>) -> tensor<256xi64>
        %2987:11 = stablehlo.while(%iterArg_344 = %iterArg_200, %iterArg_345 = %iterArg_199, %iterArg_346 = %iterArg_185, %iterArg_347 = %iterArg_186, %iterArg_348 = %iterArg_187, %iterArg_349 = %iterArg_188, %iterArg_350 = %iterArg_189, %iterArg_351 = %iterArg_198, %iterArg_352 = %2957, %iterArg_353 = %2985, %iterArg_354 = %2986) : tensor<256xf64>, tensor<256x12xf64>, tensor<192x12xf64>, tensor<256x192xf64>, tensor<256xf64>, tensor<256x192xf64>, tensor<256x192xf64>, tensor<256xf64>, tensor<256x12xf64>, tensor<256xf64>, tensor<256xi64>
        cond {
          %3027 = stablehlo.is_finite %iterArg_353 : (tensor<256xf64>) -> tensor<256xi1>
          %3028 = stablehlo.not %3027 : tensor<256xi1>
          %cst_355 = stablehlo.constant dense<1.0000000001> : tensor<f64>
          %3029 = stablehlo.broadcast_in_dim %cst_355, dims = [] : (tensor<f64>) -> tensor<256xf64>
          %3030 = stablehlo.multiply %iterArg_344, %3029 : tensor<256xf64>
          %cst_356 = stablehlo.constant dense<1.000000e-10> : tensor<f64>
          %3031 = stablehlo.broadcast_in_dim %cst_356, dims = [] : (tensor<f64>) -> tensor<256xf64>
          %3032 = stablehlo.add %3030, %3031 : tensor<256xf64>
          %3033 = stablehlo.compare GT, %iterArg_353, %3032, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
          %3034 = stablehlo.or %3028, %3033 : tensor<256xi1>
          %c_357 = stablehlo.constant dense<30> : tensor<i64>
          %3035 = stablehlo.broadcast_in_dim %c_357, dims = [] : (tensor<i64>) -> tensor<256xi64>
          %3036 = stablehlo.compare LT, %iterArg_354, %3035, SIGNED : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
          %c_358 = stablehlo.constant dense<false> : tensor<i1>
          %3037 = stablehlo.broadcast_in_dim %c_358, dims = [] : (tensor<i1>) -> tensor<256xi1>
          %3038 = stablehlo.compare NE, %3036, %3037, UNSIGNED : (tensor<256xi1>, tensor<256xi1>) -> tensor<256xi1>
          %3039 = stablehlo.convert %3038 : tensor<256xi1>
          %3040 = stablehlo.and %3034, %3039 : tensor<256xi1>
          %c_359 = stablehlo.constant dense<false> : tensor<i1>
          %3041 = stablehlo.reduce(%3040 init: %c_359) applies stablehlo.or across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
          stablehlo.return %3041 : tensor<i1>
        } do {
          %3027 = stablehlo.add %iterArg_352, %iterArg_345 : tensor<256x12xf64>
          %cst_355 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
          %3028 = stablehlo.broadcast_in_dim %cst_355, dims = [] : (tensor<f64>) -> tensor<256x12xf64>
          %3029 = stablehlo.multiply %3028, %3027 : tensor<256x12xf64>
          %3030 = stablehlo.dot_general %iterArg_346, %3029, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<192x12xf64>, tensor<256x12xf64>) -> tensor<192x256xf64>
          %3031 = stablehlo.transpose %3030, dims = [1, 0] : (tensor<192x256xf64>) -> tensor<256x192xf64>
          %3032 = stablehlo.add %3031, %iterArg_347 : tensor<256x192xf64>
          %3033 = stablehlo.exponential %3032 : tensor<256x192xf64>
          %cst_356 = stablehlo.constant dense<1.000000e-300> : tensor<f64>
          %3034 = stablehlo.broadcast_in_dim %cst_356, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
          %3035 = stablehlo.maximum %3033, %3034 : tensor<256x192xf64>
          %cst_357 = stablehlo.constant dense<-690.77552789821368> : tensor<f64>
          %cst_358 = stablehlo.constant dense<690.77552789821368> : tensor<f64>
          %3036 = func.call @clip(%3032, %cst_357, %cst_358) : (tensor<256x192xf64>, tensor<f64>, tensor<f64>) -> tensor<256x192xf64>
          %cst_359 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
          %3037 = stablehlo.broadcast_in_dim %cst_359, dims = [] : (tensor<f64>) -> tensor<256xf64>
          %3038 = stablehlo.compare GT, %iterArg_348, %3037, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
          %cst_360 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
          %3039 = func.call @_where_3(%3038, %iterArg_348, %cst_360) : (tensor<256xi1>, tensor<256xf64>, tensor<f64>) -> tensor<256xf64>
          %cst_361 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
          %3040 = stablehlo.broadcast_in_dim %cst_361, dims = [] : (tensor<f64>) -> tensor<256xf64>
          %3041 = stablehlo.divide %3040, %3039 : tensor<256xf64>
          %3042 = stablehlo.broadcast_in_dim %3041, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
          %3043 = stablehlo.broadcast_in_dim %3042, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
          %3044 = stablehlo.add %iterArg_349, %3043 : tensor<256x192xf64>
          %3045 = stablehlo.broadcast_in_dim %iterArg_348, dims = [0] : (tensor<256xf64>) -> tensor<256x1xf64>
          %3046 = stablehlo.broadcast_in_dim %3045, dims = [0, 1] : (tensor<256x1xf64>) -> tensor<256x192xf64>
          %3047 = stablehlo.multiply %3046, %3035 : tensor<256x192xf64>
          %3048 = stablehlo.log_plus_one %3047 : tensor<256x192xf64>
          %3049 = stablehlo.multiply %3044, %3048 : tensor<256x192xf64>
          %3050 = func.call @_where_5(%3038, %3049, %3035) : (tensor<256xi1>, tensor<256x192xf64>, tensor<256x192xf64>) -> tensor<256x192xf64>
          %3051 = stablehlo.multiply %iterArg_349, %3036 : tensor<256x192xf64>
          %3052 = stablehlo.subtract %3050, %3051 : tensor<256x192xf64>
          %3053 = stablehlo.multiply %iterArg_350, %3052 : tensor<256x192xf64>
          %cst_362 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
          %3054 = stablehlo.reduce(%3053 init: %cst_362) applies stablehlo.add across dimensions = [1] : (tensor<256x192xf64>, tensor<f64>) -> tensor<256xf64>
          %3055 = stablehlo.add %iterArg_351, %3054 : tensor<256xf64>
          %cst_363 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
          %3056 = stablehlo.broadcast_in_dim %cst_363, dims = [] : (tensor<f64>) -> tensor<256xf64>
          %3057 = stablehlo.multiply %3056, %3055 : tensor<256xf64>
          %c_364 = stablehlo.constant dense<1> : tensor<i64>
          %3058 = stablehlo.broadcast_in_dim %c_364, dims = [] : (tensor<i64>) -> tensor<256xi64>
          %3059 = stablehlo.add %iterArg_354, %3058 : tensor<256xi64>
          %3060 = stablehlo.is_finite %iterArg_353 : (tensor<256xf64>) -> tensor<256xi1>
          %3061 = stablehlo.not %3060 : tensor<256xi1>
          %cst_365 = stablehlo.constant dense<1.0000000001> : tensor<f64>
          %3062 = stablehlo.broadcast_in_dim %cst_365, dims = [] : (tensor<f64>) -> tensor<256xf64>
          %3063 = stablehlo.multiply %iterArg_344, %3062 : tensor<256xf64>
          %cst_366 = stablehlo.constant dense<1.000000e-10> : tensor<f64>
          %3064 = stablehlo.broadcast_in_dim %cst_366, dims = [] : (tensor<f64>) -> tensor<256xf64>
          %3065 = stablehlo.add %3063, %3064 : tensor<256xf64>
          %3066 = stablehlo.compare GT, %iterArg_353, %3065, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
          %3067 = stablehlo.or %3061, %3066 : tensor<256xi1>
          %c_367 = stablehlo.constant dense<30> : tensor<i64>
          %3068 = stablehlo.broadcast_in_dim %c_367, dims = [] : (tensor<i64>) -> tensor<256xi64>
          %3069 = stablehlo.compare LT, %iterArg_354, %3068, SIGNED : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
          %c_368 = stablehlo.constant dense<false> : tensor<i1>
          %3070 = stablehlo.broadcast_in_dim %c_368, dims = [] : (tensor<i1>) -> tensor<256xi1>
          %3071 = stablehlo.compare NE, %3069, %3070, UNSIGNED : (tensor<256xi1>, tensor<256xi1>) -> tensor<256xi1>
          %3072 = stablehlo.convert %3071 : tensor<256xi1>
          %3073 = stablehlo.and %3067, %3072 : tensor<256xi1>
          %3074 = stablehlo.broadcast_in_dim %3073, dims = [0] : (tensor<256xi1>) -> tensor<256x12xi1>
          %3075 = stablehlo.select %3074, %3029, %iterArg_352 : tensor<256x12xi1>, tensor<256x12xf64>
          %3076 = stablehlo.broadcast_in_dim %3073, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
          %3077 = stablehlo.select %3076, %3057, %iterArg_353 : tensor<256xi1>, tensor<256xf64>
          %3078 = stablehlo.broadcast_in_dim %3073, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
          %3079 = stablehlo.select %3078, %3059, %iterArg_354 : tensor<256xi1>, tensor<256xi64>
          stablehlo.return %iterArg_344, %iterArg_345, %iterArg_346, %iterArg_347, %iterArg_348, %iterArg_349, %iterArg_350, %iterArg_351, %3075, %3077, %3079 : tensor<256xf64>, tensor<256x12xf64>, tensor<192x12xf64>, tensor<256x192xf64>, tensor<256xf64>, tensor<256x192xf64>, tensor<256x192xf64>, tensor<256xf64>, tensor<256x12xf64>, tensor<256xf64>, tensor<256xi64>
        }
        %2988 = stablehlo.subtract %iterArg_200, %2987#9 : tensor<256xf64>
        %2989 = stablehlo.abs %2988 : tensor<256xf64>
        %2990 = stablehlo.abs %2987#9 : tensor<256xf64>
        %cst_334 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
        %2991 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2992 = stablehlo.add %2990, %2991 : tensor<256xf64>
        %2993 = stablehlo.divide %2989, %2992 : tensor<256xf64>
        %cst_335 = stablehlo.constant dense<1.000000e-08> : tensor<f64>
        %2994 = stablehlo.broadcast_in_dim %cst_335, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %2995 = stablehlo.compare LT, %2993, %2994, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
        %2996 = stablehlo.subtract %2987#8, %iterArg_199 : tensor<256x12xf64>
        %2997 = stablehlo.abs %2996 : tensor<256x12xf64>
        %cst_336 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
        %2998 = stablehlo.reduce(%2997 init: %cst_336) applies stablehlo.maximum across dimensions = [1] : (tensor<256x12xf64>, tensor<f64>) -> tensor<256xf64>
        %2999 = stablehlo.abs %2987#8 : tensor<256x12xf64>
        %cst_337 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
        %3000 = stablehlo.reduce(%2999 init: %cst_337) applies stablehlo.maximum across dimensions = [1] : (tensor<256x12xf64>, tensor<f64>) -> tensor<256xf64>
        %cst_338 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %3001 = stablehlo.broadcast_in_dim %cst_338, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %3002 = stablehlo.add %3001, %3000 : tensor<256xf64>
        %cst_339 = stablehlo.constant dense<1.000000e-08> : tensor<f64>
        %3003 = stablehlo.broadcast_in_dim %cst_339, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %3004 = stablehlo.multiply %3003, %3002 : tensor<256xf64>
        %3005 = stablehlo.compare LT, %2998, %3004, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
        %cst_340 = stablehlo.constant dense<9.9999999999999995E-7> : tensor<f64>
        %3006 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<f64>) -> tensor<256xf64>
        %3007 = stablehlo.compare LT, %2987#9, %3006, FLOAT : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
        %3008 = stablehlo.or %3005, %3007 : tensor<256xi1>
        %3009 = stablehlo.and %2995, %3008 : tensor<256xi1>
        %c_341 = stablehlo.constant dense<1> : tensor<i64>
        %3010 = stablehlo.broadcast_in_dim %c_341, dims = [] : (tensor<i64>) -> tensor<256xi64>
        %3011 = stablehlo.add %iterArg_201, %3010 : tensor<256xi64>
        %c_342 = stablehlo.constant dense<50> : tensor<i64>
        %3012 = stablehlo.broadcast_in_dim %c_342, dims = [] : (tensor<i64>) -> tensor<256xi64>
        %3013 = stablehlo.compare LT, %iterArg_201, %3012, SIGNED : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
        %3014 = stablehlo.not %iterArg_202 : tensor<256xi1>
        %c_343 = stablehlo.constant dense<false> : tensor<i1>
        %3015 = stablehlo.broadcast_in_dim %c_343, dims = [] : (tensor<i1>) -> tensor<256xi1>
        %3016 = stablehlo.compare NE, %3013, %3015, UNSIGNED : (tensor<256xi1>, tensor<256xi1>) -> tensor<256xi1>
        %3017 = stablehlo.convert %3016 : tensor<256xi1>
        %3018 = stablehlo.and %3017, %3014 : tensor<256xi1>
        %3019 = stablehlo.broadcast_in_dim %3018, dims = [0] : (tensor<256xi1>) -> tensor<256x12xi1>
        %3020 = stablehlo.select %3019, %2987#8, %iterArg_199 : tensor<256x12xi1>, tensor<256x12xf64>
        %3021 = stablehlo.broadcast_in_dim %3018, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
        %3022 = stablehlo.select %3021, %2987#9, %iterArg_200 : tensor<256xi1>, tensor<256xf64>
        %3023 = stablehlo.broadcast_in_dim %3018, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
        %3024 = stablehlo.select %3023, %3011, %iterArg_201 : tensor<256xi1>, tensor<256xi64>
        %3025 = stablehlo.broadcast_in_dim %3018, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
        %3026 = stablehlo.select %3025, %3009, %iterArg_202 : tensor<256xi1>, tensor<256xi1>
        stablehlo.return %iterArg_185, %iterArg_186, %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %3020, %3022, %3024, %3026 : tensor<192x12xf64>, tensor<256x192xf64>, tensor<256xf64>, tensor<256x192xf64>, tensor<256x192xf64>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<256xf64>, tensor<256x12xf64>, tensor<256xf64>, tensor<256xi64>, tensor<256xi1>
      }
      %1500 = func.call @dynamic_update_index_in_dim(%iterArg_49, %1499#14, %iterArg_48) : (tensor<1x256x12xf64>, tensor<256x12xf64>, tensor<i64>) -> tensor<1x256x12xf64>
      %1501 = func.call @dynamic_update_index_in_dim_7(%iterArg_50, %1499#15, %iterArg_48) : (tensor<1x256xf64>, tensor<256xf64>, tensor<i64>) -> tensor<1x256xf64>
      %1502 = func.call @dynamic_update_index_in_dim_8(%iterArg_51, %1499#16, %iterArg_48) : (tensor<1x256xi64>, tensor<256xi64>, tensor<i64>) -> tensor<1x256xi64>
      %1503 = func.call @dynamic_update_index_in_dim_9(%iterArg_52, %1499#17, %iterArg_48) : (tensor<1x256xi1>, tensor<256xi1>, tensor<i64>) -> tensor<1x256xi1>
      %c_184 = stablehlo.constant dense<1> : tensor<i64>
      %1504 = stablehlo.add %iterArg_48, %c_184 : tensor<i64>
      stablehlo.return %iterArg, %iterArg_27, %iterArg_28, %iterArg_29, %iterArg_30, %iterArg_31, %iterArg_32, %iterArg_33, %iterArg_34, %iterArg_35, %iterArg_36, %iterArg_37, %iterArg_38, %iterArg_39, %iterArg_40, %iterArg_41, %iterArg_42, %iterArg_43, %iterArg_44, %iterArg_45, %iterArg_46, %iterArg_47, %1504, %1500, %1501, %1502, %1503 : tensor<1x256x192xf64>, tensor<1x256x192xf64>, tensor<1x256x192xf64>, tensor<1x256xf64>, tensor<1x256x12xf64>, tensor<192x12xf64>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<i64>, tensor<1x256x12xf64>, tensor<1x256xf64>, tensor<1x256xi64>, tensor<1x256xi1>
    }
    %10 = stablehlo.reshape %9#23 : (tensor<1x256x12xf64>) -> tensor<256x12xf64>
    %11 = stablehlo.reshape %9#24 : (tensor<1x256xf64>) -> tensor<256xf64>
    %12 = stablehlo.reshape %9#25 : (tensor<1x256xi64>) -> tensor<256xi64>
    %13 = stablehlo.reshape %9#26 : (tensor<1x256xi1>) -> tensor<256xi1>
    return %10, %11, %12, %13 : tensor<256x12xf64>, tensor<256xf64>, tensor<256xi64>, tensor<256xi1>
  }
  func.func private @dynamic_index_in_dim(%arg0: tensor<1x256x192xf64>, %arg1: tensor<i64>) -> tensor<256x192xf64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c_0, sizes = [1, 256, 192] : (tensor<1x256x192xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x192xf64>
    %1 = stablehlo.reshape %0 : (tensor<1x256x192xf64>) -> tensor<256x192xf64>
    return %1 : tensor<256x192xf64>
  }
  func.func private @dynamic_index_in_dim_0(%arg0: tensor<1x256xf64>, %arg1: tensor<i64>) -> tensor<256xf64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, sizes = [1, 256] : (tensor<1x256xf64>, tensor<i64>, tensor<i64>) -> tensor<1x256xf64>
    %1 = stablehlo.reshape %0 : (tensor<1x256xf64>) -> tensor<256xf64>
    return %1 : tensor<256xf64>
  }
  func.func private @dynamic_index_in_dim_1(%arg0: tensor<1x256x12xf64>, %arg1: tensor<i64>) -> tensor<256x12xf64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c_0, sizes = [1, 256, 12] : (tensor<1x256x12xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x12xf64>
    %1 = stablehlo.reshape %0 : (tensor<1x256x12xf64>) -> tensor<256x12xf64>
    return %1 : tensor<256x12xf64>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<256x192xf64>, %arg2: tensor<256x192xf64>) -> tensor<256x192xf64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<256x192xf64>
    return %0 : tensor<256x192xf64>
  }
  func.func private @_where_2(%arg0: tensor<i1>, %arg1: tensor<256x12xf64>, %arg2: tensor<12x256xf64>) -> tensor<256x12xf64> {
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<12x256xf64>) -> tensor<256x12xf64>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<256x12xf64>
    return %1 : tensor<256x12xf64>
  }
  func.func private @_where_3(%arg0: tensor<256xi1>, %arg1: tensor<256xf64>, %arg2: tensor<f64>) -> tensor<256xf64> {
    %0 = stablehlo.convert %arg2 : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<256xf64>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<256xi1>, tensor<256xf64>
    return %2 : tensor<256xf64>
  }
  func.func private @_where_4(%arg0: tensor<256x192xi1>, %arg1: tensor<256x192xf64>, %arg2: tensor<192xf64>) -> tensor<256x192xf64> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<192xf64>) -> tensor<256x192xf64>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<256x192xi1>, tensor<256x192xf64>
    return %1 : tensor<256x192xf64>
  }
  func.func private @_where_5(%arg0: tensor<256xi1>, %arg1: tensor<256x192xf64>, %arg2: tensor<256x192xf64>) -> tensor<256x192xf64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<256xi1>) -> tensor<256x192xi1>
    %1 = stablehlo.select %0, %arg1, %arg2 : tensor<256x192xi1>, tensor<256x192xf64>
    return %1 : tensor<256x192xf64>
  }
  func.func private @clip(%arg0: tensor<256x192xf64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<256x192xf64> {
    %0 = stablehlo.convert %arg1 : tensor<f64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
    %2 = stablehlo.maximum %1, %arg0 : tensor<256x192xf64>
    %3 = stablehlo.convert %arg2 : tensor<f64>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f64>) -> tensor<256x192xf64>
    %5 = stablehlo.minimum %4, %2 : tensor<256x192xf64>
    return %5 : tensor<256x192xf64>
  }
  func.func private @_where_6(%arg0: tensor<256x192xi1>, %arg1: tensor<256x192xf64>, %arg2: tensor<256x192xf64>) -> tensor<256x192xf64> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<256x192xi1>, tensor<256x192xf64>
    return %0 : tensor<256x192xf64>
  }
  func.func private @dynamic_update_index_in_dim(%arg0: tensor<1x256x12xf64>, %arg1: tensor<256x12xf64>, %arg2: tensor<i64>) -> tensor<1x256x12xf64> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<256x12xf64>) -> tensor<1x256x12xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c_0 : (tensor<1x256x12xf64>, tensor<1x256x12xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x12xf64>
    return %1 : tensor<1x256x12xf64>
  }
  func.func private @dynamic_update_index_in_dim_7(%arg0: tensor<1x256xf64>, %arg1: tensor<256xf64>, %arg2: tensor<i64>) -> tensor<1x256xf64> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf64>) -> tensor<1x256xf64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c : (tensor<1x256xf64>, tensor<1x256xf64>, tensor<i64>, tensor<i64>) -> tensor<1x256xf64>
    return %1 : tensor<1x256xf64>
  }
  func.func private @dynamic_update_index_in_dim_8(%arg0: tensor<1x256xi64>, %arg1: tensor<256xi64>, %arg2: tensor<i64>) -> tensor<1x256xi64> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xi64>) -> tensor<1x256xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c : (tensor<1x256xi64>, tensor<1x256xi64>, tensor<i64>, tensor<i64>) -> tensor<1x256xi64>
    return %1 : tensor<1x256xi64>
  }
  func.func private @dynamic_update_index_in_dim_9(%arg0: tensor<1x256xi1>, %arg1: tensor<256xi1>, %arg2: tensor<i64>) -> tensor<1x256xi1> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xi1>) -> tensor<1x256xi1>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c : (tensor<1x256xi1>, tensor<1x256xi1>, tensor<i64>, tensor<i64>) -> tensor<1x256xi1>
    return %1 : tensor<1x256xi1>
  }
}
