module @jit_nb_fit attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x8xf32>, %arg1: tensor<8x12xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<256xf32>) -> (tensor<256x12xf32> {jax.result_info = "result[0]"}, tensor<256xf32> {jax.result_info = "result[1]"}, tensor<256xi1> {jax.result_info = "result[2]"}, tensor<256xi32> {jax.result_info = "result[3]"}) {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<8xf32>) -> tensor<256x8xf32>
    %1 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<8xf32>) -> tensor<256x8xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<256x12xf32>
    %3:4 = call @_fit_batched(%arg0, %arg1, %0, %1, %arg4, %2) : (tensor<256x8xf32>, tensor<8x12xf32>, tensor<256x8xf32>, tensor<256x8xf32>, tensor<256xf32>, tensor<256x12xf32>) -> (tensor<256x12xf32>, tensor<256xf32>, tensor<256xi64>, tensor<256xi1>)
    %4 = stablehlo.convert %3#2 : (tensor<256xi64>) -> tensor<256xi32>
    return %3#0, %3#1, %3#3, %4 : tensor<256x12xf32>, tensor<256xf32>, tensor<256xi1>, tensor<256xi32>
  }
  func.func private @_fit_batched(%arg0: tensor<256x8xf32>, %arg1: tensor<8x12xf32>, %arg2: tensor<256x8xf32>, %arg3: tensor<256x8xf32>, %arg4: tensor<256xf32>, %arg5: tensor<256x12xf32>) -> (tensor<256x12xf32>, tensor<256xf32>, tensor<256xi64>, tensor<256xi1>) {
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
    %0 = stablehlo.reshape %arg0 : (tensor<256x8xf32>) -> tensor<1x256x8xf32>
    %1 = stablehlo.reshape %arg2 : (tensor<256x8xf32>) -> tensor<1x256x8xf32>
    %2 = stablehlo.reshape %arg3 : (tensor<256x8xf32>) -> tensor<1x256x8xf32>
    %3 = stablehlo.reshape %arg4 : (tensor<256xf32>) -> tensor<1x256xf32>
    %4 = stablehlo.reshape %arg5 : (tensor<256x12xf32>) -> tensor<1x256x12xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x256x12xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x256xf32>
    %c_24 = stablehlo.constant dense<0> : tensor<i64>
    %7 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i64>) -> tensor<1x256xi64>
    %c_25 = stablehlo.constant dense<false> : tensor<i1>
    %8 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i1>) -> tensor<1x256xi1>
    %c_26 = stablehlo.constant dense<0> : tensor<i64>
    %9:27 = stablehlo.while(%iterArg = %0, %iterArg_27 = %1, %iterArg_28 = %2, %iterArg_29 = %3, %iterArg_30 = %4, %iterArg_31 = %arg1, %iterArg_32 = %c, %iterArg_33 = %c_0, %iterArg_34 = %c_1, %iterArg_35 = %c_2, %iterArg_36 = %c_3, %iterArg_37 = %c_4, %iterArg_38 = %c_5, %iterArg_39 = %c_6, %iterArg_40 = %c_7, %iterArg_41 = %c_8, %iterArg_42 = %c_9, %iterArg_43 = %c_10, %iterArg_44 = %c_11, %iterArg_45 = %c_12, %iterArg_46 = %c_13, %iterArg_47 = %c_14, %iterArg_48 = %c_26, %iterArg_49 = %5, %iterArg_50 = %6, %iterArg_51 = %7, %iterArg_52 = %8) : tensor<1x256x8xf32>, tensor<1x256x8xf32>, tensor<1x256x8xf32>, tensor<1x256xf32>, tensor<1x256x12xf32>, tensor<8x12xf32>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<i64>, tensor<1x256x12xf32>, tensor<1x256xf32>, tensor<1x256xi64>, tensor<1x256xi1>
    cond {
      %c_53 = stablehlo.constant dense<1> : tensor<i64>
      %14 = stablehlo.compare LT, %iterArg_48, %c_53, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14 = func.call @dynamic_index_in_dim(%iterArg, %iterArg_48) : (tensor<1x256x8xf32>, tensor<i64>) -> tensor<256x8xf32>
      %15 = func.call @dynamic_index_in_dim(%iterArg_27, %iterArg_48) : (tensor<1x256x8xf32>, tensor<i64>) -> tensor<256x8xf32>
      %16 = func.call @dynamic_index_in_dim(%iterArg_28, %iterArg_48) : (tensor<1x256x8xf32>, tensor<i64>) -> tensor<256x8xf32>
      %17 = func.call @dynamic_index_in_dim_0(%iterArg_29, %iterArg_48) : (tensor<1x256xf32>, tensor<i64>) -> tensor<256xf32>
      %18 = func.call @dynamic_index_in_dim_1(%iterArg_30, %iterArg_48) : (tensor<1x256x12xf32>, tensor<i64>) -> tensor<256x12xf32>
      %cst_53 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
      %19 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %20 = stablehlo.add %14, %19 : tensor<256x8xf32>
      %21 = stablehlo.log %20 : tensor<256x8xf32>
      %22 = stablehlo.dot_general %iterArg_31, %18, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<8x12xf32>, tensor<256x12xf32>) -> tensor<8x256xf32>
      %23 = stablehlo.transpose %22, dims = [1, 0] : (tensor<8x256xf32>) -> tensor<256x8xf32>
      %24 = stablehlo.add %23, %15 : tensor<256x8xf32>
      %c_54 = stablehlo.constant dense<false> : tensor<i1>
      %25 = func.call @_where(%c_54, %24, %21) : (tensor<i1>, tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xf32>
      %26 = stablehlo.exponential %25 : tensor<256x8xf32>
      %cst_55 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %27 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %28 = stablehlo.maximum %26, %27 : tensor<256x8xf32>
      %29 = stablehlo.exponential %25 : tensor<256x8xf32>
      %30 = stablehlo.multiply %28, %28 : tensor<256x8xf32>
      %31 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
      %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
      %33 = stablehlo.multiply %32, %30 : tensor<256x8xf32>
      %34 = stablehlo.add %28, %33 : tensor<256x8xf32>
      %35 = stablehlo.multiply %29, %29 : tensor<256x8xf32>
      %36 = stablehlo.multiply %16, %35 : tensor<256x8xf32>
      %37 = stablehlo.divide %36, %34 : tensor<256x8xf32>
      %38 = stablehlo.subtract %14, %28 : tensor<256x8xf32>
      %39 = stablehlo.divide %38, %29 : tensor<256x8xf32>
      %40 = stablehlo.add %25, %39 : tensor<256x8xf32>
      %41 = stablehlo.subtract %40, %15 : tensor<256x8xf32>
      %c_56 = stablehlo.constant dense<12> : tensor<i64>
      %42 = stablehlo.broadcast_in_dim %c_56, dims = [] : (tensor<i64>) -> tensor<78xi64>
      %43 = stablehlo.add %iterArg_32, %42 : tensor<78xi64>
      %44 = stablehlo.select %iterArg_33, %43, %iterArg_32 : tensor<78xi1>, tensor<78xi64>
      %45 = stablehlo.convert %44 : (tensor<78xi64>) -> tensor<78xi32>
      %46 = stablehlo.broadcast_in_dim %45, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
      %47 = "stablehlo.gather"(%iterArg_31, %46) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x12xf32>, tensor<78x1xi32>) -> tensor<8x78xf32>
      %c_57 = stablehlo.constant dense<12> : tensor<i64>
      %48 = stablehlo.broadcast_in_dim %c_57, dims = [] : (tensor<i64>) -> tensor<78xi64>
      %49 = stablehlo.add %iterArg_34, %48 : tensor<78xi64>
      %50 = stablehlo.select %iterArg_35, %49, %iterArg_34 : tensor<78xi1>, tensor<78xi64>
      %51 = stablehlo.convert %50 : (tensor<78xi64>) -> tensor<78xi32>
      %52 = stablehlo.broadcast_in_dim %51, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
      %53 = "stablehlo.gather"(%iterArg_31, %52) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x12xf32>, tensor<78x1xi32>) -> tensor<8x78xf32>
      %54 = stablehlo.multiply %47, %53 : tensor<8x78xf32>
      %55 = stablehlo.dot_general %37, %54, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x8xf32>, tensor<8x78xf32>) -> tensor<256x78xf32>
      %cst_58 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %56 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f32>) -> tensor<12x12xf32>
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
      %68 = stablehlo.broadcast_in_dim %56, dims = [1, 2] : (tensor<12x12xf32>) -> tensor<256x12x12xf32>
      %69 = "stablehlo.scatter"(%68, %67, %55) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        stablehlo.return %arg7 : tensor<f32>
      }) : (tensor<256x12x12xf32>, tensor<78x2xi32>, tensor<256x78xf32>) -> tensor<256x12x12xf32>
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
      ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
        stablehlo.return %arg7 : tensor<f32>
      }) : (tensor<256x12x12xf32>, tensor<78x2xi32>, tensor<256x78xf32>) -> tensor<256x12x12xf32>
      %82 = stablehlo.iota dim = 0 : tensor<12x12xi64>
      %83 = stablehlo.iota dim = 1 : tensor<12x12xi64>
      %c_63 = stablehlo.constant dense<0> : tensor<i64>
      %84 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i64>) -> tensor<12x12xi64>
      %85 = stablehlo.add %82, %84 : tensor<12x12xi64>
      %86 = stablehlo.compare EQ, %85, %83, SIGNED : (tensor<12x12xi64>, tensor<12x12xi64>) -> tensor<12x12xi1>
      %87 = stablehlo.convert %86 : (tensor<12x12xi1>) -> tensor<12x12xf32>
      %cst_64 = stablehlo.constant dense<1.000000e-10> : tensor<f32>
      %88 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f32>) -> tensor<12x12xf32>
      %89 = stablehlo.multiply %88, %87 : tensor<12x12xf32>
      %90 = stablehlo.broadcast_in_dim %89, dims = [1, 2] : (tensor<12x12xf32>) -> tensor<1x12x12xf32>
      %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<1x12x12xf32>) -> tensor<256x12x12xf32>
      %92 = stablehlo.add %81, %91 : tensor<256x12x12xf32>
      %93 = stablehlo.transpose %iterArg_31, dims = [1, 0] : (tensor<8x12xf32>) -> tensor<12x8xf32>
      %94 = stablehlo.multiply %37, %41 : tensor<256x8xf32>
      %95 = stablehlo.dot_general %93, %94, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<12x8xf32>, tensor<256x8xf32>) -> tensor<12x256xf32>
      %96 = stablehlo.slice %92 [0:256, 0:1, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %97 = stablehlo.reshape %96 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %98 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %99 = stablehlo.subtract %97, %98 : tensor<256xf32>
      %100 = stablehlo.sqrt %99 : tensor<256xf32>
      %101 = stablehlo.slice %92 [0:256, 1:2, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %102 = stablehlo.reshape %101 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_66 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %103 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %104 = stablehlo.subtract %102, %103 : tensor<256xf32>
      %105 = stablehlo.divide %104, %100 : tensor<256xf32>
      %106 = stablehlo.slice %92 [0:256, 2:3, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %107 = stablehlo.reshape %106 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_67 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %108 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %109 = stablehlo.subtract %107, %108 : tensor<256xf32>
      %110 = stablehlo.divide %109, %100 : tensor<256xf32>
      %111 = stablehlo.slice %92 [0:256, 3:4, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %112 = stablehlo.reshape %111 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_68 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %113 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %114 = stablehlo.subtract %112, %113 : tensor<256xf32>
      %115 = stablehlo.divide %114, %100 : tensor<256xf32>
      %116 = stablehlo.slice %92 [0:256, 4:5, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %117 = stablehlo.reshape %116 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_69 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %118 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %119 = stablehlo.subtract %117, %118 : tensor<256xf32>
      %120 = stablehlo.divide %119, %100 : tensor<256xf32>
      %121 = stablehlo.slice %92 [0:256, 5:6, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %122 = stablehlo.reshape %121 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_70 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %123 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %124 = stablehlo.subtract %122, %123 : tensor<256xf32>
      %125 = stablehlo.divide %124, %100 : tensor<256xf32>
      %126 = stablehlo.slice %92 [0:256, 6:7, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %127 = stablehlo.reshape %126 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_71 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %128 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %129 = stablehlo.subtract %127, %128 : tensor<256xf32>
      %130 = stablehlo.divide %129, %100 : tensor<256xf32>
      %131 = stablehlo.slice %92 [0:256, 7:8, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %132 = stablehlo.reshape %131 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_72 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %133 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %134 = stablehlo.subtract %132, %133 : tensor<256xf32>
      %135 = stablehlo.divide %134, %100 : tensor<256xf32>
      %136 = stablehlo.slice %92 [0:256, 8:9, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %137 = stablehlo.reshape %136 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_73 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %138 = stablehlo.broadcast_in_dim %cst_73, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %139 = stablehlo.subtract %137, %138 : tensor<256xf32>
      %140 = stablehlo.divide %139, %100 : tensor<256xf32>
      %141 = stablehlo.slice %92 [0:256, 9:10, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %142 = stablehlo.reshape %141 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_74 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %143 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %144 = stablehlo.subtract %142, %143 : tensor<256xf32>
      %145 = stablehlo.divide %144, %100 : tensor<256xf32>
      %146 = stablehlo.slice %92 [0:256, 10:11, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %147 = stablehlo.reshape %146 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_75 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %148 = stablehlo.broadcast_in_dim %cst_75, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %149 = stablehlo.subtract %147, %148 : tensor<256xf32>
      %150 = stablehlo.divide %149, %100 : tensor<256xf32>
      %151 = stablehlo.slice %92 [0:256, 11:12, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %152 = stablehlo.reshape %151 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %cst_76 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %153 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %154 = stablehlo.subtract %152, %153 : tensor<256xf32>
      %155 = stablehlo.divide %154, %100 : tensor<256xf32>
      %156 = stablehlo.slice %92 [0:256, 1:2, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %157 = stablehlo.reshape %156 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %158 = stablehlo.multiply %105, %105 : tensor<256xf32>
      %cst_77 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %159 = stablehlo.broadcast_in_dim %cst_77, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %160 = stablehlo.add %159, %158 : tensor<256xf32>
      %161 = stablehlo.subtract %157, %160 : tensor<256xf32>
      %162 = stablehlo.sqrt %161 : tensor<256xf32>
      %163 = stablehlo.slice %92 [0:256, 2:3, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %164 = stablehlo.reshape %163 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %165 = stablehlo.multiply %110, %105 : tensor<256xf32>
      %cst_78 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %166 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %167 = stablehlo.add %166, %165 : tensor<256xf32>
      %168 = stablehlo.subtract %164, %167 : tensor<256xf32>
      %169 = stablehlo.divide %168, %162 : tensor<256xf32>
      %170 = stablehlo.slice %92 [0:256, 3:4, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %171 = stablehlo.reshape %170 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %172 = stablehlo.multiply %115, %105 : tensor<256xf32>
      %cst_79 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %173 = stablehlo.broadcast_in_dim %cst_79, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %174 = stablehlo.add %173, %172 : tensor<256xf32>
      %175 = stablehlo.subtract %171, %174 : tensor<256xf32>
      %176 = stablehlo.divide %175, %162 : tensor<256xf32>
      %177 = stablehlo.slice %92 [0:256, 4:5, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %178 = stablehlo.reshape %177 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %179 = stablehlo.multiply %120, %105 : tensor<256xf32>
      %cst_80 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %180 = stablehlo.broadcast_in_dim %cst_80, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %181 = stablehlo.add %180, %179 : tensor<256xf32>
      %182 = stablehlo.subtract %178, %181 : tensor<256xf32>
      %183 = stablehlo.divide %182, %162 : tensor<256xf32>
      %184 = stablehlo.slice %92 [0:256, 5:6, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %185 = stablehlo.reshape %184 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %186 = stablehlo.multiply %125, %105 : tensor<256xf32>
      %cst_81 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %187 = stablehlo.broadcast_in_dim %cst_81, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %188 = stablehlo.add %187, %186 : tensor<256xf32>
      %189 = stablehlo.subtract %185, %188 : tensor<256xf32>
      %190 = stablehlo.divide %189, %162 : tensor<256xf32>
      %191 = stablehlo.slice %92 [0:256, 6:7, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %192 = stablehlo.reshape %191 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %193 = stablehlo.multiply %130, %105 : tensor<256xf32>
      %cst_82 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %194 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %195 = stablehlo.add %194, %193 : tensor<256xf32>
      %196 = stablehlo.subtract %192, %195 : tensor<256xf32>
      %197 = stablehlo.divide %196, %162 : tensor<256xf32>
      %198 = stablehlo.slice %92 [0:256, 7:8, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %199 = stablehlo.reshape %198 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %200 = stablehlo.multiply %135, %105 : tensor<256xf32>
      %cst_83 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %201 = stablehlo.broadcast_in_dim %cst_83, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %202 = stablehlo.add %201, %200 : tensor<256xf32>
      %203 = stablehlo.subtract %199, %202 : tensor<256xf32>
      %204 = stablehlo.divide %203, %162 : tensor<256xf32>
      %205 = stablehlo.slice %92 [0:256, 8:9, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %206 = stablehlo.reshape %205 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %207 = stablehlo.multiply %140, %105 : tensor<256xf32>
      %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %208 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %209 = stablehlo.add %208, %207 : tensor<256xf32>
      %210 = stablehlo.subtract %206, %209 : tensor<256xf32>
      %211 = stablehlo.divide %210, %162 : tensor<256xf32>
      %212 = stablehlo.slice %92 [0:256, 9:10, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %213 = stablehlo.reshape %212 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %214 = stablehlo.multiply %145, %105 : tensor<256xf32>
      %cst_85 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %215 = stablehlo.broadcast_in_dim %cst_85, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %216 = stablehlo.add %215, %214 : tensor<256xf32>
      %217 = stablehlo.subtract %213, %216 : tensor<256xf32>
      %218 = stablehlo.divide %217, %162 : tensor<256xf32>
      %219 = stablehlo.slice %92 [0:256, 10:11, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %220 = stablehlo.reshape %219 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %221 = stablehlo.multiply %150, %105 : tensor<256xf32>
      %cst_86 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %222 = stablehlo.broadcast_in_dim %cst_86, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %223 = stablehlo.add %222, %221 : tensor<256xf32>
      %224 = stablehlo.subtract %220, %223 : tensor<256xf32>
      %225 = stablehlo.divide %224, %162 : tensor<256xf32>
      %226 = stablehlo.slice %92 [0:256, 11:12, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %227 = stablehlo.reshape %226 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %228 = stablehlo.multiply %155, %105 : tensor<256xf32>
      %cst_87 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %229 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %230 = stablehlo.add %229, %228 : tensor<256xf32>
      %231 = stablehlo.subtract %227, %230 : tensor<256xf32>
      %232 = stablehlo.divide %231, %162 : tensor<256xf32>
      %233 = stablehlo.slice %92 [0:256, 2:3, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %234 = stablehlo.reshape %233 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %235 = stablehlo.multiply %110, %110 : tensor<256xf32>
      %cst_88 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %236 = stablehlo.broadcast_in_dim %cst_88, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %237 = stablehlo.add %236, %235 : tensor<256xf32>
      %238 = stablehlo.multiply %169, %169 : tensor<256xf32>
      %239 = stablehlo.add %237, %238 : tensor<256xf32>
      %240 = stablehlo.subtract %234, %239 : tensor<256xf32>
      %241 = stablehlo.sqrt %240 : tensor<256xf32>
      %242 = stablehlo.slice %92 [0:256, 3:4, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %243 = stablehlo.reshape %242 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %244 = stablehlo.multiply %115, %110 : tensor<256xf32>
      %cst_89 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %245 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %246 = stablehlo.add %245, %244 : tensor<256xf32>
      %247 = stablehlo.multiply %176, %169 : tensor<256xf32>
      %248 = stablehlo.add %246, %247 : tensor<256xf32>
      %249 = stablehlo.subtract %243, %248 : tensor<256xf32>
      %250 = stablehlo.divide %249, %241 : tensor<256xf32>
      %251 = stablehlo.slice %92 [0:256, 4:5, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %252 = stablehlo.reshape %251 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %253 = stablehlo.multiply %120, %110 : tensor<256xf32>
      %cst_90 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %254 = stablehlo.broadcast_in_dim %cst_90, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %255 = stablehlo.add %254, %253 : tensor<256xf32>
      %256 = stablehlo.multiply %183, %169 : tensor<256xf32>
      %257 = stablehlo.add %255, %256 : tensor<256xf32>
      %258 = stablehlo.subtract %252, %257 : tensor<256xf32>
      %259 = stablehlo.divide %258, %241 : tensor<256xf32>
      %260 = stablehlo.slice %92 [0:256, 5:6, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %261 = stablehlo.reshape %260 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %262 = stablehlo.multiply %125, %110 : tensor<256xf32>
      %cst_91 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %263 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %264 = stablehlo.add %263, %262 : tensor<256xf32>
      %265 = stablehlo.multiply %190, %169 : tensor<256xf32>
      %266 = stablehlo.add %264, %265 : tensor<256xf32>
      %267 = stablehlo.subtract %261, %266 : tensor<256xf32>
      %268 = stablehlo.divide %267, %241 : tensor<256xf32>
      %269 = stablehlo.slice %92 [0:256, 6:7, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %270 = stablehlo.reshape %269 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %271 = stablehlo.multiply %130, %110 : tensor<256xf32>
      %cst_92 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %272 = stablehlo.broadcast_in_dim %cst_92, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %273 = stablehlo.add %272, %271 : tensor<256xf32>
      %274 = stablehlo.multiply %197, %169 : tensor<256xf32>
      %275 = stablehlo.add %273, %274 : tensor<256xf32>
      %276 = stablehlo.subtract %270, %275 : tensor<256xf32>
      %277 = stablehlo.divide %276, %241 : tensor<256xf32>
      %278 = stablehlo.slice %92 [0:256, 7:8, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %279 = stablehlo.reshape %278 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %280 = stablehlo.multiply %135, %110 : tensor<256xf32>
      %cst_93 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %281 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %282 = stablehlo.add %281, %280 : tensor<256xf32>
      %283 = stablehlo.multiply %204, %169 : tensor<256xf32>
      %284 = stablehlo.add %282, %283 : tensor<256xf32>
      %285 = stablehlo.subtract %279, %284 : tensor<256xf32>
      %286 = stablehlo.divide %285, %241 : tensor<256xf32>
      %287 = stablehlo.slice %92 [0:256, 8:9, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %288 = stablehlo.reshape %287 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %289 = stablehlo.multiply %140, %110 : tensor<256xf32>
      %cst_94 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %290 = stablehlo.broadcast_in_dim %cst_94, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %291 = stablehlo.add %290, %289 : tensor<256xf32>
      %292 = stablehlo.multiply %211, %169 : tensor<256xf32>
      %293 = stablehlo.add %291, %292 : tensor<256xf32>
      %294 = stablehlo.subtract %288, %293 : tensor<256xf32>
      %295 = stablehlo.divide %294, %241 : tensor<256xf32>
      %296 = stablehlo.slice %92 [0:256, 9:10, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %297 = stablehlo.reshape %296 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %298 = stablehlo.multiply %145, %110 : tensor<256xf32>
      %cst_95 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %299 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %300 = stablehlo.add %299, %298 : tensor<256xf32>
      %301 = stablehlo.multiply %218, %169 : tensor<256xf32>
      %302 = stablehlo.add %300, %301 : tensor<256xf32>
      %303 = stablehlo.subtract %297, %302 : tensor<256xf32>
      %304 = stablehlo.divide %303, %241 : tensor<256xf32>
      %305 = stablehlo.slice %92 [0:256, 10:11, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %306 = stablehlo.reshape %305 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %307 = stablehlo.multiply %150, %110 : tensor<256xf32>
      %cst_96 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %308 = stablehlo.broadcast_in_dim %cst_96, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %309 = stablehlo.add %308, %307 : tensor<256xf32>
      %310 = stablehlo.multiply %225, %169 : tensor<256xf32>
      %311 = stablehlo.add %309, %310 : tensor<256xf32>
      %312 = stablehlo.subtract %306, %311 : tensor<256xf32>
      %313 = stablehlo.divide %312, %241 : tensor<256xf32>
      %314 = stablehlo.slice %92 [0:256, 11:12, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %315 = stablehlo.reshape %314 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %316 = stablehlo.multiply %155, %110 : tensor<256xf32>
      %cst_97 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %317 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %318 = stablehlo.add %317, %316 : tensor<256xf32>
      %319 = stablehlo.multiply %232, %169 : tensor<256xf32>
      %320 = stablehlo.add %318, %319 : tensor<256xf32>
      %321 = stablehlo.subtract %315, %320 : tensor<256xf32>
      %322 = stablehlo.divide %321, %241 : tensor<256xf32>
      %323 = stablehlo.slice %92 [0:256, 3:4, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %324 = stablehlo.reshape %323 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %325 = stablehlo.multiply %115, %115 : tensor<256xf32>
      %cst_98 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %326 = stablehlo.broadcast_in_dim %cst_98, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %327 = stablehlo.add %326, %325 : tensor<256xf32>
      %328 = stablehlo.multiply %176, %176 : tensor<256xf32>
      %329 = stablehlo.add %327, %328 : tensor<256xf32>
      %330 = stablehlo.multiply %250, %250 : tensor<256xf32>
      %331 = stablehlo.add %329, %330 : tensor<256xf32>
      %332 = stablehlo.subtract %324, %331 : tensor<256xf32>
      %333 = stablehlo.sqrt %332 : tensor<256xf32>
      %334 = stablehlo.slice %92 [0:256, 4:5, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %335 = stablehlo.reshape %334 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %336 = stablehlo.multiply %120, %115 : tensor<256xf32>
      %cst_99 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %337 = stablehlo.broadcast_in_dim %cst_99, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %338 = stablehlo.add %337, %336 : tensor<256xf32>
      %339 = stablehlo.multiply %183, %176 : tensor<256xf32>
      %340 = stablehlo.add %338, %339 : tensor<256xf32>
      %341 = stablehlo.multiply %259, %250 : tensor<256xf32>
      %342 = stablehlo.add %340, %341 : tensor<256xf32>
      %343 = stablehlo.subtract %335, %342 : tensor<256xf32>
      %344 = stablehlo.divide %343, %333 : tensor<256xf32>
      %345 = stablehlo.slice %92 [0:256, 5:6, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %346 = stablehlo.reshape %345 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %347 = stablehlo.multiply %125, %115 : tensor<256xf32>
      %cst_100 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %348 = stablehlo.broadcast_in_dim %cst_100, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %349 = stablehlo.add %348, %347 : tensor<256xf32>
      %350 = stablehlo.multiply %190, %176 : tensor<256xf32>
      %351 = stablehlo.add %349, %350 : tensor<256xf32>
      %352 = stablehlo.multiply %268, %250 : tensor<256xf32>
      %353 = stablehlo.add %351, %352 : tensor<256xf32>
      %354 = stablehlo.subtract %346, %353 : tensor<256xf32>
      %355 = stablehlo.divide %354, %333 : tensor<256xf32>
      %356 = stablehlo.slice %92 [0:256, 6:7, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %357 = stablehlo.reshape %356 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %358 = stablehlo.multiply %130, %115 : tensor<256xf32>
      %cst_101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %359 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %360 = stablehlo.add %359, %358 : tensor<256xf32>
      %361 = stablehlo.multiply %197, %176 : tensor<256xf32>
      %362 = stablehlo.add %360, %361 : tensor<256xf32>
      %363 = stablehlo.multiply %277, %250 : tensor<256xf32>
      %364 = stablehlo.add %362, %363 : tensor<256xf32>
      %365 = stablehlo.subtract %357, %364 : tensor<256xf32>
      %366 = stablehlo.divide %365, %333 : tensor<256xf32>
      %367 = stablehlo.slice %92 [0:256, 7:8, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %368 = stablehlo.reshape %367 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %369 = stablehlo.multiply %135, %115 : tensor<256xf32>
      %cst_102 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %370 = stablehlo.broadcast_in_dim %cst_102, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %371 = stablehlo.add %370, %369 : tensor<256xf32>
      %372 = stablehlo.multiply %204, %176 : tensor<256xf32>
      %373 = stablehlo.add %371, %372 : tensor<256xf32>
      %374 = stablehlo.multiply %286, %250 : tensor<256xf32>
      %375 = stablehlo.add %373, %374 : tensor<256xf32>
      %376 = stablehlo.subtract %368, %375 : tensor<256xf32>
      %377 = stablehlo.divide %376, %333 : tensor<256xf32>
      %378 = stablehlo.slice %92 [0:256, 8:9, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %379 = stablehlo.reshape %378 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %380 = stablehlo.multiply %140, %115 : tensor<256xf32>
      %cst_103 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %381 = stablehlo.broadcast_in_dim %cst_103, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %382 = stablehlo.add %381, %380 : tensor<256xf32>
      %383 = stablehlo.multiply %211, %176 : tensor<256xf32>
      %384 = stablehlo.add %382, %383 : tensor<256xf32>
      %385 = stablehlo.multiply %295, %250 : tensor<256xf32>
      %386 = stablehlo.add %384, %385 : tensor<256xf32>
      %387 = stablehlo.subtract %379, %386 : tensor<256xf32>
      %388 = stablehlo.divide %387, %333 : tensor<256xf32>
      %389 = stablehlo.slice %92 [0:256, 9:10, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %390 = stablehlo.reshape %389 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %391 = stablehlo.multiply %145, %115 : tensor<256xf32>
      %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %392 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %393 = stablehlo.add %392, %391 : tensor<256xf32>
      %394 = stablehlo.multiply %218, %176 : tensor<256xf32>
      %395 = stablehlo.add %393, %394 : tensor<256xf32>
      %396 = stablehlo.multiply %304, %250 : tensor<256xf32>
      %397 = stablehlo.add %395, %396 : tensor<256xf32>
      %398 = stablehlo.subtract %390, %397 : tensor<256xf32>
      %399 = stablehlo.divide %398, %333 : tensor<256xf32>
      %400 = stablehlo.slice %92 [0:256, 10:11, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %401 = stablehlo.reshape %400 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %402 = stablehlo.multiply %150, %115 : tensor<256xf32>
      %cst_105 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %403 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %404 = stablehlo.add %403, %402 : tensor<256xf32>
      %405 = stablehlo.multiply %225, %176 : tensor<256xf32>
      %406 = stablehlo.add %404, %405 : tensor<256xf32>
      %407 = stablehlo.multiply %313, %250 : tensor<256xf32>
      %408 = stablehlo.add %406, %407 : tensor<256xf32>
      %409 = stablehlo.subtract %401, %408 : tensor<256xf32>
      %410 = stablehlo.divide %409, %333 : tensor<256xf32>
      %411 = stablehlo.slice %92 [0:256, 11:12, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %412 = stablehlo.reshape %411 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %413 = stablehlo.multiply %155, %115 : tensor<256xf32>
      %cst_106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %414 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %415 = stablehlo.add %414, %413 : tensor<256xf32>
      %416 = stablehlo.multiply %232, %176 : tensor<256xf32>
      %417 = stablehlo.add %415, %416 : tensor<256xf32>
      %418 = stablehlo.multiply %322, %250 : tensor<256xf32>
      %419 = stablehlo.add %417, %418 : tensor<256xf32>
      %420 = stablehlo.subtract %412, %419 : tensor<256xf32>
      %421 = stablehlo.divide %420, %333 : tensor<256xf32>
      %422 = stablehlo.slice %92 [0:256, 4:5, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %423 = stablehlo.reshape %422 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %424 = stablehlo.multiply %120, %120 : tensor<256xf32>
      %cst_107 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %425 = stablehlo.broadcast_in_dim %cst_107, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %426 = stablehlo.add %425, %424 : tensor<256xf32>
      %427 = stablehlo.multiply %183, %183 : tensor<256xf32>
      %428 = stablehlo.add %426, %427 : tensor<256xf32>
      %429 = stablehlo.multiply %259, %259 : tensor<256xf32>
      %430 = stablehlo.add %428, %429 : tensor<256xf32>
      %431 = stablehlo.multiply %344, %344 : tensor<256xf32>
      %432 = stablehlo.add %430, %431 : tensor<256xf32>
      %433 = stablehlo.subtract %423, %432 : tensor<256xf32>
      %434 = stablehlo.sqrt %433 : tensor<256xf32>
      %435 = stablehlo.slice %92 [0:256, 5:6, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %436 = stablehlo.reshape %435 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %437 = stablehlo.multiply %125, %120 : tensor<256xf32>
      %cst_108 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %438 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %439 = stablehlo.add %438, %437 : tensor<256xf32>
      %440 = stablehlo.multiply %190, %183 : tensor<256xf32>
      %441 = stablehlo.add %439, %440 : tensor<256xf32>
      %442 = stablehlo.multiply %268, %259 : tensor<256xf32>
      %443 = stablehlo.add %441, %442 : tensor<256xf32>
      %444 = stablehlo.multiply %355, %344 : tensor<256xf32>
      %445 = stablehlo.add %443, %444 : tensor<256xf32>
      %446 = stablehlo.subtract %436, %445 : tensor<256xf32>
      %447 = stablehlo.divide %446, %434 : tensor<256xf32>
      %448 = stablehlo.slice %92 [0:256, 6:7, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %449 = stablehlo.reshape %448 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %450 = stablehlo.multiply %130, %120 : tensor<256xf32>
      %cst_109 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %451 = stablehlo.broadcast_in_dim %cst_109, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %452 = stablehlo.add %451, %450 : tensor<256xf32>
      %453 = stablehlo.multiply %197, %183 : tensor<256xf32>
      %454 = stablehlo.add %452, %453 : tensor<256xf32>
      %455 = stablehlo.multiply %277, %259 : tensor<256xf32>
      %456 = stablehlo.add %454, %455 : tensor<256xf32>
      %457 = stablehlo.multiply %366, %344 : tensor<256xf32>
      %458 = stablehlo.add %456, %457 : tensor<256xf32>
      %459 = stablehlo.subtract %449, %458 : tensor<256xf32>
      %460 = stablehlo.divide %459, %434 : tensor<256xf32>
      %461 = stablehlo.slice %92 [0:256, 7:8, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %462 = stablehlo.reshape %461 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %463 = stablehlo.multiply %135, %120 : tensor<256xf32>
      %cst_110 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %464 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %465 = stablehlo.add %464, %463 : tensor<256xf32>
      %466 = stablehlo.multiply %204, %183 : tensor<256xf32>
      %467 = stablehlo.add %465, %466 : tensor<256xf32>
      %468 = stablehlo.multiply %286, %259 : tensor<256xf32>
      %469 = stablehlo.add %467, %468 : tensor<256xf32>
      %470 = stablehlo.multiply %377, %344 : tensor<256xf32>
      %471 = stablehlo.add %469, %470 : tensor<256xf32>
      %472 = stablehlo.subtract %462, %471 : tensor<256xf32>
      %473 = stablehlo.divide %472, %434 : tensor<256xf32>
      %474 = stablehlo.slice %92 [0:256, 8:9, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %475 = stablehlo.reshape %474 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %476 = stablehlo.multiply %140, %120 : tensor<256xf32>
      %cst_111 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %477 = stablehlo.broadcast_in_dim %cst_111, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %478 = stablehlo.add %477, %476 : tensor<256xf32>
      %479 = stablehlo.multiply %211, %183 : tensor<256xf32>
      %480 = stablehlo.add %478, %479 : tensor<256xf32>
      %481 = stablehlo.multiply %295, %259 : tensor<256xf32>
      %482 = stablehlo.add %480, %481 : tensor<256xf32>
      %483 = stablehlo.multiply %388, %344 : tensor<256xf32>
      %484 = stablehlo.add %482, %483 : tensor<256xf32>
      %485 = stablehlo.subtract %475, %484 : tensor<256xf32>
      %486 = stablehlo.divide %485, %434 : tensor<256xf32>
      %487 = stablehlo.slice %92 [0:256, 9:10, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %488 = stablehlo.reshape %487 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %489 = stablehlo.multiply %145, %120 : tensor<256xf32>
      %cst_112 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %490 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %491 = stablehlo.add %490, %489 : tensor<256xf32>
      %492 = stablehlo.multiply %218, %183 : tensor<256xf32>
      %493 = stablehlo.add %491, %492 : tensor<256xf32>
      %494 = stablehlo.multiply %304, %259 : tensor<256xf32>
      %495 = stablehlo.add %493, %494 : tensor<256xf32>
      %496 = stablehlo.multiply %399, %344 : tensor<256xf32>
      %497 = stablehlo.add %495, %496 : tensor<256xf32>
      %498 = stablehlo.subtract %488, %497 : tensor<256xf32>
      %499 = stablehlo.divide %498, %434 : tensor<256xf32>
      %500 = stablehlo.slice %92 [0:256, 10:11, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %501 = stablehlo.reshape %500 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %502 = stablehlo.multiply %150, %120 : tensor<256xf32>
      %cst_113 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %503 = stablehlo.broadcast_in_dim %cst_113, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %504 = stablehlo.add %503, %502 : tensor<256xf32>
      %505 = stablehlo.multiply %225, %183 : tensor<256xf32>
      %506 = stablehlo.add %504, %505 : tensor<256xf32>
      %507 = stablehlo.multiply %313, %259 : tensor<256xf32>
      %508 = stablehlo.add %506, %507 : tensor<256xf32>
      %509 = stablehlo.multiply %410, %344 : tensor<256xf32>
      %510 = stablehlo.add %508, %509 : tensor<256xf32>
      %511 = stablehlo.subtract %501, %510 : tensor<256xf32>
      %512 = stablehlo.divide %511, %434 : tensor<256xf32>
      %513 = stablehlo.slice %92 [0:256, 11:12, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %514 = stablehlo.reshape %513 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %515 = stablehlo.multiply %155, %120 : tensor<256xf32>
      %cst_114 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %516 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %517 = stablehlo.add %516, %515 : tensor<256xf32>
      %518 = stablehlo.multiply %232, %183 : tensor<256xf32>
      %519 = stablehlo.add %517, %518 : tensor<256xf32>
      %520 = stablehlo.multiply %322, %259 : tensor<256xf32>
      %521 = stablehlo.add %519, %520 : tensor<256xf32>
      %522 = stablehlo.multiply %421, %344 : tensor<256xf32>
      %523 = stablehlo.add %521, %522 : tensor<256xf32>
      %524 = stablehlo.subtract %514, %523 : tensor<256xf32>
      %525 = stablehlo.divide %524, %434 : tensor<256xf32>
      %526 = stablehlo.slice %92 [0:256, 5:6, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %527 = stablehlo.reshape %526 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %528 = stablehlo.multiply %125, %125 : tensor<256xf32>
      %cst_115 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %529 = stablehlo.broadcast_in_dim %cst_115, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %530 = stablehlo.add %529, %528 : tensor<256xf32>
      %531 = stablehlo.multiply %190, %190 : tensor<256xf32>
      %532 = stablehlo.add %530, %531 : tensor<256xf32>
      %533 = stablehlo.multiply %268, %268 : tensor<256xf32>
      %534 = stablehlo.add %532, %533 : tensor<256xf32>
      %535 = stablehlo.multiply %355, %355 : tensor<256xf32>
      %536 = stablehlo.add %534, %535 : tensor<256xf32>
      %537 = stablehlo.multiply %447, %447 : tensor<256xf32>
      %538 = stablehlo.add %536, %537 : tensor<256xf32>
      %539 = stablehlo.subtract %527, %538 : tensor<256xf32>
      %540 = stablehlo.sqrt %539 : tensor<256xf32>
      %541 = stablehlo.slice %92 [0:256, 6:7, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %542 = stablehlo.reshape %541 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %543 = stablehlo.multiply %130, %125 : tensor<256xf32>
      %cst_116 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %544 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %545 = stablehlo.add %544, %543 : tensor<256xf32>
      %546 = stablehlo.multiply %197, %190 : tensor<256xf32>
      %547 = stablehlo.add %545, %546 : tensor<256xf32>
      %548 = stablehlo.multiply %277, %268 : tensor<256xf32>
      %549 = stablehlo.add %547, %548 : tensor<256xf32>
      %550 = stablehlo.multiply %366, %355 : tensor<256xf32>
      %551 = stablehlo.add %549, %550 : tensor<256xf32>
      %552 = stablehlo.multiply %460, %447 : tensor<256xf32>
      %553 = stablehlo.add %551, %552 : tensor<256xf32>
      %554 = stablehlo.subtract %542, %553 : tensor<256xf32>
      %555 = stablehlo.divide %554, %540 : tensor<256xf32>
      %556 = stablehlo.slice %92 [0:256, 7:8, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %557 = stablehlo.reshape %556 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %558 = stablehlo.multiply %135, %125 : tensor<256xf32>
      %cst_117 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %559 = stablehlo.broadcast_in_dim %cst_117, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %560 = stablehlo.add %559, %558 : tensor<256xf32>
      %561 = stablehlo.multiply %204, %190 : tensor<256xf32>
      %562 = stablehlo.add %560, %561 : tensor<256xf32>
      %563 = stablehlo.multiply %286, %268 : tensor<256xf32>
      %564 = stablehlo.add %562, %563 : tensor<256xf32>
      %565 = stablehlo.multiply %377, %355 : tensor<256xf32>
      %566 = stablehlo.add %564, %565 : tensor<256xf32>
      %567 = stablehlo.multiply %473, %447 : tensor<256xf32>
      %568 = stablehlo.add %566, %567 : tensor<256xf32>
      %569 = stablehlo.subtract %557, %568 : tensor<256xf32>
      %570 = stablehlo.divide %569, %540 : tensor<256xf32>
      %571 = stablehlo.slice %92 [0:256, 8:9, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %572 = stablehlo.reshape %571 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %573 = stablehlo.multiply %140, %125 : tensor<256xf32>
      %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %574 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %575 = stablehlo.add %574, %573 : tensor<256xf32>
      %576 = stablehlo.multiply %211, %190 : tensor<256xf32>
      %577 = stablehlo.add %575, %576 : tensor<256xf32>
      %578 = stablehlo.multiply %295, %268 : tensor<256xf32>
      %579 = stablehlo.add %577, %578 : tensor<256xf32>
      %580 = stablehlo.multiply %388, %355 : tensor<256xf32>
      %581 = stablehlo.add %579, %580 : tensor<256xf32>
      %582 = stablehlo.multiply %486, %447 : tensor<256xf32>
      %583 = stablehlo.add %581, %582 : tensor<256xf32>
      %584 = stablehlo.subtract %572, %583 : tensor<256xf32>
      %585 = stablehlo.divide %584, %540 : tensor<256xf32>
      %586 = stablehlo.slice %92 [0:256, 9:10, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %587 = stablehlo.reshape %586 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %588 = stablehlo.multiply %145, %125 : tensor<256xf32>
      %cst_119 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %589 = stablehlo.broadcast_in_dim %cst_119, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %590 = stablehlo.add %589, %588 : tensor<256xf32>
      %591 = stablehlo.multiply %218, %190 : tensor<256xf32>
      %592 = stablehlo.add %590, %591 : tensor<256xf32>
      %593 = stablehlo.multiply %304, %268 : tensor<256xf32>
      %594 = stablehlo.add %592, %593 : tensor<256xf32>
      %595 = stablehlo.multiply %399, %355 : tensor<256xf32>
      %596 = stablehlo.add %594, %595 : tensor<256xf32>
      %597 = stablehlo.multiply %499, %447 : tensor<256xf32>
      %598 = stablehlo.add %596, %597 : tensor<256xf32>
      %599 = stablehlo.subtract %587, %598 : tensor<256xf32>
      %600 = stablehlo.divide %599, %540 : tensor<256xf32>
      %601 = stablehlo.slice %92 [0:256, 10:11, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %602 = stablehlo.reshape %601 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %603 = stablehlo.multiply %150, %125 : tensor<256xf32>
      %cst_120 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %604 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %605 = stablehlo.add %604, %603 : tensor<256xf32>
      %606 = stablehlo.multiply %225, %190 : tensor<256xf32>
      %607 = stablehlo.add %605, %606 : tensor<256xf32>
      %608 = stablehlo.multiply %313, %268 : tensor<256xf32>
      %609 = stablehlo.add %607, %608 : tensor<256xf32>
      %610 = stablehlo.multiply %410, %355 : tensor<256xf32>
      %611 = stablehlo.add %609, %610 : tensor<256xf32>
      %612 = stablehlo.multiply %512, %447 : tensor<256xf32>
      %613 = stablehlo.add %611, %612 : tensor<256xf32>
      %614 = stablehlo.subtract %602, %613 : tensor<256xf32>
      %615 = stablehlo.divide %614, %540 : tensor<256xf32>
      %616 = stablehlo.slice %92 [0:256, 11:12, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %617 = stablehlo.reshape %616 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %618 = stablehlo.multiply %155, %125 : tensor<256xf32>
      %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %619 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %620 = stablehlo.add %619, %618 : tensor<256xf32>
      %621 = stablehlo.multiply %232, %190 : tensor<256xf32>
      %622 = stablehlo.add %620, %621 : tensor<256xf32>
      %623 = stablehlo.multiply %322, %268 : tensor<256xf32>
      %624 = stablehlo.add %622, %623 : tensor<256xf32>
      %625 = stablehlo.multiply %421, %355 : tensor<256xf32>
      %626 = stablehlo.add %624, %625 : tensor<256xf32>
      %627 = stablehlo.multiply %525, %447 : tensor<256xf32>
      %628 = stablehlo.add %626, %627 : tensor<256xf32>
      %629 = stablehlo.subtract %617, %628 : tensor<256xf32>
      %630 = stablehlo.divide %629, %540 : tensor<256xf32>
      %631 = stablehlo.slice %92 [0:256, 6:7, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %632 = stablehlo.reshape %631 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %633 = stablehlo.multiply %130, %130 : tensor<256xf32>
      %cst_122 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %634 = stablehlo.broadcast_in_dim %cst_122, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %635 = stablehlo.add %634, %633 : tensor<256xf32>
      %636 = stablehlo.multiply %197, %197 : tensor<256xf32>
      %637 = stablehlo.add %635, %636 : tensor<256xf32>
      %638 = stablehlo.multiply %277, %277 : tensor<256xf32>
      %639 = stablehlo.add %637, %638 : tensor<256xf32>
      %640 = stablehlo.multiply %366, %366 : tensor<256xf32>
      %641 = stablehlo.add %639, %640 : tensor<256xf32>
      %642 = stablehlo.multiply %460, %460 : tensor<256xf32>
      %643 = stablehlo.add %641, %642 : tensor<256xf32>
      %644 = stablehlo.multiply %555, %555 : tensor<256xf32>
      %645 = stablehlo.add %643, %644 : tensor<256xf32>
      %646 = stablehlo.subtract %632, %645 : tensor<256xf32>
      %647 = stablehlo.sqrt %646 : tensor<256xf32>
      %648 = stablehlo.slice %92 [0:256, 7:8, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %649 = stablehlo.reshape %648 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %650 = stablehlo.multiply %135, %130 : tensor<256xf32>
      %cst_123 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %651 = stablehlo.broadcast_in_dim %cst_123, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %652 = stablehlo.add %651, %650 : tensor<256xf32>
      %653 = stablehlo.multiply %204, %197 : tensor<256xf32>
      %654 = stablehlo.add %652, %653 : tensor<256xf32>
      %655 = stablehlo.multiply %286, %277 : tensor<256xf32>
      %656 = stablehlo.add %654, %655 : tensor<256xf32>
      %657 = stablehlo.multiply %377, %366 : tensor<256xf32>
      %658 = stablehlo.add %656, %657 : tensor<256xf32>
      %659 = stablehlo.multiply %473, %460 : tensor<256xf32>
      %660 = stablehlo.add %658, %659 : tensor<256xf32>
      %661 = stablehlo.multiply %570, %555 : tensor<256xf32>
      %662 = stablehlo.add %660, %661 : tensor<256xf32>
      %663 = stablehlo.subtract %649, %662 : tensor<256xf32>
      %664 = stablehlo.divide %663, %647 : tensor<256xf32>
      %665 = stablehlo.slice %92 [0:256, 8:9, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %666 = stablehlo.reshape %665 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %667 = stablehlo.multiply %140, %130 : tensor<256xf32>
      %cst_124 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %668 = stablehlo.broadcast_in_dim %cst_124, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %669 = stablehlo.add %668, %667 : tensor<256xf32>
      %670 = stablehlo.multiply %211, %197 : tensor<256xf32>
      %671 = stablehlo.add %669, %670 : tensor<256xf32>
      %672 = stablehlo.multiply %295, %277 : tensor<256xf32>
      %673 = stablehlo.add %671, %672 : tensor<256xf32>
      %674 = stablehlo.multiply %388, %366 : tensor<256xf32>
      %675 = stablehlo.add %673, %674 : tensor<256xf32>
      %676 = stablehlo.multiply %486, %460 : tensor<256xf32>
      %677 = stablehlo.add %675, %676 : tensor<256xf32>
      %678 = stablehlo.multiply %585, %555 : tensor<256xf32>
      %679 = stablehlo.add %677, %678 : tensor<256xf32>
      %680 = stablehlo.subtract %666, %679 : tensor<256xf32>
      %681 = stablehlo.divide %680, %647 : tensor<256xf32>
      %682 = stablehlo.slice %92 [0:256, 9:10, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %683 = stablehlo.reshape %682 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %684 = stablehlo.multiply %145, %130 : tensor<256xf32>
      %cst_125 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %685 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %686 = stablehlo.add %685, %684 : tensor<256xf32>
      %687 = stablehlo.multiply %218, %197 : tensor<256xf32>
      %688 = stablehlo.add %686, %687 : tensor<256xf32>
      %689 = stablehlo.multiply %304, %277 : tensor<256xf32>
      %690 = stablehlo.add %688, %689 : tensor<256xf32>
      %691 = stablehlo.multiply %399, %366 : tensor<256xf32>
      %692 = stablehlo.add %690, %691 : tensor<256xf32>
      %693 = stablehlo.multiply %499, %460 : tensor<256xf32>
      %694 = stablehlo.add %692, %693 : tensor<256xf32>
      %695 = stablehlo.multiply %600, %555 : tensor<256xf32>
      %696 = stablehlo.add %694, %695 : tensor<256xf32>
      %697 = stablehlo.subtract %683, %696 : tensor<256xf32>
      %698 = stablehlo.divide %697, %647 : tensor<256xf32>
      %699 = stablehlo.slice %92 [0:256, 10:11, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %700 = stablehlo.reshape %699 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %701 = stablehlo.multiply %150, %130 : tensor<256xf32>
      %cst_126 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %702 = stablehlo.broadcast_in_dim %cst_126, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %703 = stablehlo.add %702, %701 : tensor<256xf32>
      %704 = stablehlo.multiply %225, %197 : tensor<256xf32>
      %705 = stablehlo.add %703, %704 : tensor<256xf32>
      %706 = stablehlo.multiply %313, %277 : tensor<256xf32>
      %707 = stablehlo.add %705, %706 : tensor<256xf32>
      %708 = stablehlo.multiply %410, %366 : tensor<256xf32>
      %709 = stablehlo.add %707, %708 : tensor<256xf32>
      %710 = stablehlo.multiply %512, %460 : tensor<256xf32>
      %711 = stablehlo.add %709, %710 : tensor<256xf32>
      %712 = stablehlo.multiply %615, %555 : tensor<256xf32>
      %713 = stablehlo.add %711, %712 : tensor<256xf32>
      %714 = stablehlo.subtract %700, %713 : tensor<256xf32>
      %715 = stablehlo.divide %714, %647 : tensor<256xf32>
      %716 = stablehlo.slice %92 [0:256, 11:12, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %717 = stablehlo.reshape %716 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %718 = stablehlo.multiply %155, %130 : tensor<256xf32>
      %cst_127 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %719 = stablehlo.broadcast_in_dim %cst_127, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %720 = stablehlo.add %719, %718 : tensor<256xf32>
      %721 = stablehlo.multiply %232, %197 : tensor<256xf32>
      %722 = stablehlo.add %720, %721 : tensor<256xf32>
      %723 = stablehlo.multiply %322, %277 : tensor<256xf32>
      %724 = stablehlo.add %722, %723 : tensor<256xf32>
      %725 = stablehlo.multiply %421, %366 : tensor<256xf32>
      %726 = stablehlo.add %724, %725 : tensor<256xf32>
      %727 = stablehlo.multiply %525, %460 : tensor<256xf32>
      %728 = stablehlo.add %726, %727 : tensor<256xf32>
      %729 = stablehlo.multiply %630, %555 : tensor<256xf32>
      %730 = stablehlo.add %728, %729 : tensor<256xf32>
      %731 = stablehlo.subtract %717, %730 : tensor<256xf32>
      %732 = stablehlo.divide %731, %647 : tensor<256xf32>
      %733 = stablehlo.slice %92 [0:256, 7:8, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %734 = stablehlo.reshape %733 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %735 = stablehlo.multiply %135, %135 : tensor<256xf32>
      %cst_128 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %736 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %737 = stablehlo.add %736, %735 : tensor<256xf32>
      %738 = stablehlo.multiply %204, %204 : tensor<256xf32>
      %739 = stablehlo.add %737, %738 : tensor<256xf32>
      %740 = stablehlo.multiply %286, %286 : tensor<256xf32>
      %741 = stablehlo.add %739, %740 : tensor<256xf32>
      %742 = stablehlo.multiply %377, %377 : tensor<256xf32>
      %743 = stablehlo.add %741, %742 : tensor<256xf32>
      %744 = stablehlo.multiply %473, %473 : tensor<256xf32>
      %745 = stablehlo.add %743, %744 : tensor<256xf32>
      %746 = stablehlo.multiply %570, %570 : tensor<256xf32>
      %747 = stablehlo.add %745, %746 : tensor<256xf32>
      %748 = stablehlo.multiply %664, %664 : tensor<256xf32>
      %749 = stablehlo.add %747, %748 : tensor<256xf32>
      %750 = stablehlo.subtract %734, %749 : tensor<256xf32>
      %751 = stablehlo.sqrt %750 : tensor<256xf32>
      %752 = stablehlo.slice %92 [0:256, 8:9, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %753 = stablehlo.reshape %752 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %754 = stablehlo.multiply %140, %135 : tensor<256xf32>
      %cst_129 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %755 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %756 = stablehlo.add %755, %754 : tensor<256xf32>
      %757 = stablehlo.multiply %211, %204 : tensor<256xf32>
      %758 = stablehlo.add %756, %757 : tensor<256xf32>
      %759 = stablehlo.multiply %295, %286 : tensor<256xf32>
      %760 = stablehlo.add %758, %759 : tensor<256xf32>
      %761 = stablehlo.multiply %388, %377 : tensor<256xf32>
      %762 = stablehlo.add %760, %761 : tensor<256xf32>
      %763 = stablehlo.multiply %486, %473 : tensor<256xf32>
      %764 = stablehlo.add %762, %763 : tensor<256xf32>
      %765 = stablehlo.multiply %585, %570 : tensor<256xf32>
      %766 = stablehlo.add %764, %765 : tensor<256xf32>
      %767 = stablehlo.multiply %681, %664 : tensor<256xf32>
      %768 = stablehlo.add %766, %767 : tensor<256xf32>
      %769 = stablehlo.subtract %753, %768 : tensor<256xf32>
      %770 = stablehlo.divide %769, %751 : tensor<256xf32>
      %771 = stablehlo.slice %92 [0:256, 9:10, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %772 = stablehlo.reshape %771 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %773 = stablehlo.multiply %145, %135 : tensor<256xf32>
      %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %774 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %775 = stablehlo.add %774, %773 : tensor<256xf32>
      %776 = stablehlo.multiply %218, %204 : tensor<256xf32>
      %777 = stablehlo.add %775, %776 : tensor<256xf32>
      %778 = stablehlo.multiply %304, %286 : tensor<256xf32>
      %779 = stablehlo.add %777, %778 : tensor<256xf32>
      %780 = stablehlo.multiply %399, %377 : tensor<256xf32>
      %781 = stablehlo.add %779, %780 : tensor<256xf32>
      %782 = stablehlo.multiply %499, %473 : tensor<256xf32>
      %783 = stablehlo.add %781, %782 : tensor<256xf32>
      %784 = stablehlo.multiply %600, %570 : tensor<256xf32>
      %785 = stablehlo.add %783, %784 : tensor<256xf32>
      %786 = stablehlo.multiply %698, %664 : tensor<256xf32>
      %787 = stablehlo.add %785, %786 : tensor<256xf32>
      %788 = stablehlo.subtract %772, %787 : tensor<256xf32>
      %789 = stablehlo.divide %788, %751 : tensor<256xf32>
      %790 = stablehlo.slice %92 [0:256, 10:11, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %791 = stablehlo.reshape %790 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %792 = stablehlo.multiply %150, %135 : tensor<256xf32>
      %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %793 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %794 = stablehlo.add %793, %792 : tensor<256xf32>
      %795 = stablehlo.multiply %225, %204 : tensor<256xf32>
      %796 = stablehlo.add %794, %795 : tensor<256xf32>
      %797 = stablehlo.multiply %313, %286 : tensor<256xf32>
      %798 = stablehlo.add %796, %797 : tensor<256xf32>
      %799 = stablehlo.multiply %410, %377 : tensor<256xf32>
      %800 = stablehlo.add %798, %799 : tensor<256xf32>
      %801 = stablehlo.multiply %512, %473 : tensor<256xf32>
      %802 = stablehlo.add %800, %801 : tensor<256xf32>
      %803 = stablehlo.multiply %615, %570 : tensor<256xf32>
      %804 = stablehlo.add %802, %803 : tensor<256xf32>
      %805 = stablehlo.multiply %715, %664 : tensor<256xf32>
      %806 = stablehlo.add %804, %805 : tensor<256xf32>
      %807 = stablehlo.subtract %791, %806 : tensor<256xf32>
      %808 = stablehlo.divide %807, %751 : tensor<256xf32>
      %809 = stablehlo.slice %92 [0:256, 11:12, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %810 = stablehlo.reshape %809 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %811 = stablehlo.multiply %155, %135 : tensor<256xf32>
      %cst_132 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %812 = stablehlo.broadcast_in_dim %cst_132, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %813 = stablehlo.add %812, %811 : tensor<256xf32>
      %814 = stablehlo.multiply %232, %204 : tensor<256xf32>
      %815 = stablehlo.add %813, %814 : tensor<256xf32>
      %816 = stablehlo.multiply %322, %286 : tensor<256xf32>
      %817 = stablehlo.add %815, %816 : tensor<256xf32>
      %818 = stablehlo.multiply %421, %377 : tensor<256xf32>
      %819 = stablehlo.add %817, %818 : tensor<256xf32>
      %820 = stablehlo.multiply %525, %473 : tensor<256xf32>
      %821 = stablehlo.add %819, %820 : tensor<256xf32>
      %822 = stablehlo.multiply %630, %570 : tensor<256xf32>
      %823 = stablehlo.add %821, %822 : tensor<256xf32>
      %824 = stablehlo.multiply %732, %664 : tensor<256xf32>
      %825 = stablehlo.add %823, %824 : tensor<256xf32>
      %826 = stablehlo.subtract %810, %825 : tensor<256xf32>
      %827 = stablehlo.divide %826, %751 : tensor<256xf32>
      %828 = stablehlo.slice %92 [0:256, 8:9, 8:9] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %829 = stablehlo.reshape %828 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %830 = stablehlo.multiply %140, %140 : tensor<256xf32>
      %cst_133 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %831 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %832 = stablehlo.add %831, %830 : tensor<256xf32>
      %833 = stablehlo.multiply %211, %211 : tensor<256xf32>
      %834 = stablehlo.add %832, %833 : tensor<256xf32>
      %835 = stablehlo.multiply %295, %295 : tensor<256xf32>
      %836 = stablehlo.add %834, %835 : tensor<256xf32>
      %837 = stablehlo.multiply %388, %388 : tensor<256xf32>
      %838 = stablehlo.add %836, %837 : tensor<256xf32>
      %839 = stablehlo.multiply %486, %486 : tensor<256xf32>
      %840 = stablehlo.add %838, %839 : tensor<256xf32>
      %841 = stablehlo.multiply %585, %585 : tensor<256xf32>
      %842 = stablehlo.add %840, %841 : tensor<256xf32>
      %843 = stablehlo.multiply %681, %681 : tensor<256xf32>
      %844 = stablehlo.add %842, %843 : tensor<256xf32>
      %845 = stablehlo.multiply %770, %770 : tensor<256xf32>
      %846 = stablehlo.add %844, %845 : tensor<256xf32>
      %847 = stablehlo.subtract %829, %846 : tensor<256xf32>
      %848 = stablehlo.sqrt %847 : tensor<256xf32>
      %849 = stablehlo.slice %92 [0:256, 9:10, 8:9] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %850 = stablehlo.reshape %849 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %851 = stablehlo.multiply %145, %140 : tensor<256xf32>
      %cst_134 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %852 = stablehlo.broadcast_in_dim %cst_134, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %853 = stablehlo.add %852, %851 : tensor<256xf32>
      %854 = stablehlo.multiply %218, %211 : tensor<256xf32>
      %855 = stablehlo.add %853, %854 : tensor<256xf32>
      %856 = stablehlo.multiply %304, %295 : tensor<256xf32>
      %857 = stablehlo.add %855, %856 : tensor<256xf32>
      %858 = stablehlo.multiply %399, %388 : tensor<256xf32>
      %859 = stablehlo.add %857, %858 : tensor<256xf32>
      %860 = stablehlo.multiply %499, %486 : tensor<256xf32>
      %861 = stablehlo.add %859, %860 : tensor<256xf32>
      %862 = stablehlo.multiply %600, %585 : tensor<256xf32>
      %863 = stablehlo.add %861, %862 : tensor<256xf32>
      %864 = stablehlo.multiply %698, %681 : tensor<256xf32>
      %865 = stablehlo.add %863, %864 : tensor<256xf32>
      %866 = stablehlo.multiply %789, %770 : tensor<256xf32>
      %867 = stablehlo.add %865, %866 : tensor<256xf32>
      %868 = stablehlo.subtract %850, %867 : tensor<256xf32>
      %869 = stablehlo.divide %868, %848 : tensor<256xf32>
      %870 = stablehlo.slice %92 [0:256, 10:11, 8:9] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %871 = stablehlo.reshape %870 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %872 = stablehlo.multiply %150, %140 : tensor<256xf32>
      %cst_135 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %873 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %874 = stablehlo.add %873, %872 : tensor<256xf32>
      %875 = stablehlo.multiply %225, %211 : tensor<256xf32>
      %876 = stablehlo.add %874, %875 : tensor<256xf32>
      %877 = stablehlo.multiply %313, %295 : tensor<256xf32>
      %878 = stablehlo.add %876, %877 : tensor<256xf32>
      %879 = stablehlo.multiply %410, %388 : tensor<256xf32>
      %880 = stablehlo.add %878, %879 : tensor<256xf32>
      %881 = stablehlo.multiply %512, %486 : tensor<256xf32>
      %882 = stablehlo.add %880, %881 : tensor<256xf32>
      %883 = stablehlo.multiply %615, %585 : tensor<256xf32>
      %884 = stablehlo.add %882, %883 : tensor<256xf32>
      %885 = stablehlo.multiply %715, %681 : tensor<256xf32>
      %886 = stablehlo.add %884, %885 : tensor<256xf32>
      %887 = stablehlo.multiply %808, %770 : tensor<256xf32>
      %888 = stablehlo.add %886, %887 : tensor<256xf32>
      %889 = stablehlo.subtract %871, %888 : tensor<256xf32>
      %890 = stablehlo.divide %889, %848 : tensor<256xf32>
      %891 = stablehlo.slice %92 [0:256, 11:12, 8:9] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %892 = stablehlo.reshape %891 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %893 = stablehlo.multiply %155, %140 : tensor<256xf32>
      %cst_136 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %894 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %895 = stablehlo.add %894, %893 : tensor<256xf32>
      %896 = stablehlo.multiply %232, %211 : tensor<256xf32>
      %897 = stablehlo.add %895, %896 : tensor<256xf32>
      %898 = stablehlo.multiply %322, %295 : tensor<256xf32>
      %899 = stablehlo.add %897, %898 : tensor<256xf32>
      %900 = stablehlo.multiply %421, %388 : tensor<256xf32>
      %901 = stablehlo.add %899, %900 : tensor<256xf32>
      %902 = stablehlo.multiply %525, %486 : tensor<256xf32>
      %903 = stablehlo.add %901, %902 : tensor<256xf32>
      %904 = stablehlo.multiply %630, %585 : tensor<256xf32>
      %905 = stablehlo.add %903, %904 : tensor<256xf32>
      %906 = stablehlo.multiply %732, %681 : tensor<256xf32>
      %907 = stablehlo.add %905, %906 : tensor<256xf32>
      %908 = stablehlo.multiply %827, %770 : tensor<256xf32>
      %909 = stablehlo.add %907, %908 : tensor<256xf32>
      %910 = stablehlo.subtract %892, %909 : tensor<256xf32>
      %911 = stablehlo.divide %910, %848 : tensor<256xf32>
      %912 = stablehlo.slice %92 [0:256, 9:10, 9:10] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %913 = stablehlo.reshape %912 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %914 = stablehlo.multiply %145, %145 : tensor<256xf32>
      %cst_137 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %915 = stablehlo.broadcast_in_dim %cst_137, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %916 = stablehlo.add %915, %914 : tensor<256xf32>
      %917 = stablehlo.multiply %218, %218 : tensor<256xf32>
      %918 = stablehlo.add %916, %917 : tensor<256xf32>
      %919 = stablehlo.multiply %304, %304 : tensor<256xf32>
      %920 = stablehlo.add %918, %919 : tensor<256xf32>
      %921 = stablehlo.multiply %399, %399 : tensor<256xf32>
      %922 = stablehlo.add %920, %921 : tensor<256xf32>
      %923 = stablehlo.multiply %499, %499 : tensor<256xf32>
      %924 = stablehlo.add %922, %923 : tensor<256xf32>
      %925 = stablehlo.multiply %600, %600 : tensor<256xf32>
      %926 = stablehlo.add %924, %925 : tensor<256xf32>
      %927 = stablehlo.multiply %698, %698 : tensor<256xf32>
      %928 = stablehlo.add %926, %927 : tensor<256xf32>
      %929 = stablehlo.multiply %789, %789 : tensor<256xf32>
      %930 = stablehlo.add %928, %929 : tensor<256xf32>
      %931 = stablehlo.multiply %869, %869 : tensor<256xf32>
      %932 = stablehlo.add %930, %931 : tensor<256xf32>
      %933 = stablehlo.subtract %913, %932 : tensor<256xf32>
      %934 = stablehlo.sqrt %933 : tensor<256xf32>
      %935 = stablehlo.slice %92 [0:256, 10:11, 9:10] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %936 = stablehlo.reshape %935 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %937 = stablehlo.multiply %150, %145 : tensor<256xf32>
      %cst_138 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %938 = stablehlo.broadcast_in_dim %cst_138, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %939 = stablehlo.add %938, %937 : tensor<256xf32>
      %940 = stablehlo.multiply %225, %218 : tensor<256xf32>
      %941 = stablehlo.add %939, %940 : tensor<256xf32>
      %942 = stablehlo.multiply %313, %304 : tensor<256xf32>
      %943 = stablehlo.add %941, %942 : tensor<256xf32>
      %944 = stablehlo.multiply %410, %399 : tensor<256xf32>
      %945 = stablehlo.add %943, %944 : tensor<256xf32>
      %946 = stablehlo.multiply %512, %499 : tensor<256xf32>
      %947 = stablehlo.add %945, %946 : tensor<256xf32>
      %948 = stablehlo.multiply %615, %600 : tensor<256xf32>
      %949 = stablehlo.add %947, %948 : tensor<256xf32>
      %950 = stablehlo.multiply %715, %698 : tensor<256xf32>
      %951 = stablehlo.add %949, %950 : tensor<256xf32>
      %952 = stablehlo.multiply %808, %789 : tensor<256xf32>
      %953 = stablehlo.add %951, %952 : tensor<256xf32>
      %954 = stablehlo.multiply %890, %869 : tensor<256xf32>
      %955 = stablehlo.add %953, %954 : tensor<256xf32>
      %956 = stablehlo.subtract %936, %955 : tensor<256xf32>
      %957 = stablehlo.divide %956, %934 : tensor<256xf32>
      %958 = stablehlo.slice %92 [0:256, 11:12, 9:10] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %959 = stablehlo.reshape %958 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %960 = stablehlo.multiply %155, %145 : tensor<256xf32>
      %cst_139 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %961 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %962 = stablehlo.add %961, %960 : tensor<256xf32>
      %963 = stablehlo.multiply %232, %218 : tensor<256xf32>
      %964 = stablehlo.add %962, %963 : tensor<256xf32>
      %965 = stablehlo.multiply %322, %304 : tensor<256xf32>
      %966 = stablehlo.add %964, %965 : tensor<256xf32>
      %967 = stablehlo.multiply %421, %399 : tensor<256xf32>
      %968 = stablehlo.add %966, %967 : tensor<256xf32>
      %969 = stablehlo.multiply %525, %499 : tensor<256xf32>
      %970 = stablehlo.add %968, %969 : tensor<256xf32>
      %971 = stablehlo.multiply %630, %600 : tensor<256xf32>
      %972 = stablehlo.add %970, %971 : tensor<256xf32>
      %973 = stablehlo.multiply %732, %698 : tensor<256xf32>
      %974 = stablehlo.add %972, %973 : tensor<256xf32>
      %975 = stablehlo.multiply %827, %789 : tensor<256xf32>
      %976 = stablehlo.add %974, %975 : tensor<256xf32>
      %977 = stablehlo.multiply %911, %869 : tensor<256xf32>
      %978 = stablehlo.add %976, %977 : tensor<256xf32>
      %979 = stablehlo.subtract %959, %978 : tensor<256xf32>
      %980 = stablehlo.divide %979, %934 : tensor<256xf32>
      %981 = stablehlo.slice %92 [0:256, 10:11, 10:11] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %982 = stablehlo.reshape %981 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %983 = stablehlo.multiply %150, %150 : tensor<256xf32>
      %cst_140 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %984 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %985 = stablehlo.add %984, %983 : tensor<256xf32>
      %986 = stablehlo.multiply %225, %225 : tensor<256xf32>
      %987 = stablehlo.add %985, %986 : tensor<256xf32>
      %988 = stablehlo.multiply %313, %313 : tensor<256xf32>
      %989 = stablehlo.add %987, %988 : tensor<256xf32>
      %990 = stablehlo.multiply %410, %410 : tensor<256xf32>
      %991 = stablehlo.add %989, %990 : tensor<256xf32>
      %992 = stablehlo.multiply %512, %512 : tensor<256xf32>
      %993 = stablehlo.add %991, %992 : tensor<256xf32>
      %994 = stablehlo.multiply %615, %615 : tensor<256xf32>
      %995 = stablehlo.add %993, %994 : tensor<256xf32>
      %996 = stablehlo.multiply %715, %715 : tensor<256xf32>
      %997 = stablehlo.add %995, %996 : tensor<256xf32>
      %998 = stablehlo.multiply %808, %808 : tensor<256xf32>
      %999 = stablehlo.add %997, %998 : tensor<256xf32>
      %1000 = stablehlo.multiply %890, %890 : tensor<256xf32>
      %1001 = stablehlo.add %999, %1000 : tensor<256xf32>
      %1002 = stablehlo.multiply %957, %957 : tensor<256xf32>
      %1003 = stablehlo.add %1001, %1002 : tensor<256xf32>
      %1004 = stablehlo.subtract %982, %1003 : tensor<256xf32>
      %1005 = stablehlo.sqrt %1004 : tensor<256xf32>
      %1006 = stablehlo.slice %92 [0:256, 11:12, 10:11] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %1007 = stablehlo.reshape %1006 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %1008 = stablehlo.multiply %155, %150 : tensor<256xf32>
      %cst_141 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1009 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1010 = stablehlo.add %1009, %1008 : tensor<256xf32>
      %1011 = stablehlo.multiply %232, %225 : tensor<256xf32>
      %1012 = stablehlo.add %1010, %1011 : tensor<256xf32>
      %1013 = stablehlo.multiply %322, %313 : tensor<256xf32>
      %1014 = stablehlo.add %1012, %1013 : tensor<256xf32>
      %1015 = stablehlo.multiply %421, %410 : tensor<256xf32>
      %1016 = stablehlo.add %1014, %1015 : tensor<256xf32>
      %1017 = stablehlo.multiply %525, %512 : tensor<256xf32>
      %1018 = stablehlo.add %1016, %1017 : tensor<256xf32>
      %1019 = stablehlo.multiply %630, %615 : tensor<256xf32>
      %1020 = stablehlo.add %1018, %1019 : tensor<256xf32>
      %1021 = stablehlo.multiply %732, %715 : tensor<256xf32>
      %1022 = stablehlo.add %1020, %1021 : tensor<256xf32>
      %1023 = stablehlo.multiply %827, %808 : tensor<256xf32>
      %1024 = stablehlo.add %1022, %1023 : tensor<256xf32>
      %1025 = stablehlo.multiply %911, %890 : tensor<256xf32>
      %1026 = stablehlo.add %1024, %1025 : tensor<256xf32>
      %1027 = stablehlo.multiply %980, %957 : tensor<256xf32>
      %1028 = stablehlo.add %1026, %1027 : tensor<256xf32>
      %1029 = stablehlo.subtract %1007, %1028 : tensor<256xf32>
      %1030 = stablehlo.divide %1029, %1005 : tensor<256xf32>
      %1031 = stablehlo.slice %92 [0:256, 11:12, 11:12] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
      %1032 = stablehlo.reshape %1031 : (tensor<256x1x1xf32>) -> tensor<256xf32>
      %1033 = stablehlo.multiply %155, %155 : tensor<256xf32>
      %cst_142 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1034 = stablehlo.broadcast_in_dim %cst_142, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1035 = stablehlo.add %1034, %1033 : tensor<256xf32>
      %1036 = stablehlo.multiply %232, %232 : tensor<256xf32>
      %1037 = stablehlo.add %1035, %1036 : tensor<256xf32>
      %1038 = stablehlo.multiply %322, %322 : tensor<256xf32>
      %1039 = stablehlo.add %1037, %1038 : tensor<256xf32>
      %1040 = stablehlo.multiply %421, %421 : tensor<256xf32>
      %1041 = stablehlo.add %1039, %1040 : tensor<256xf32>
      %1042 = stablehlo.multiply %525, %525 : tensor<256xf32>
      %1043 = stablehlo.add %1041, %1042 : tensor<256xf32>
      %1044 = stablehlo.multiply %630, %630 : tensor<256xf32>
      %1045 = stablehlo.add %1043, %1044 : tensor<256xf32>
      %1046 = stablehlo.multiply %732, %732 : tensor<256xf32>
      %1047 = stablehlo.add %1045, %1046 : tensor<256xf32>
      %1048 = stablehlo.multiply %827, %827 : tensor<256xf32>
      %1049 = stablehlo.add %1047, %1048 : tensor<256xf32>
      %1050 = stablehlo.multiply %911, %911 : tensor<256xf32>
      %1051 = stablehlo.add %1049, %1050 : tensor<256xf32>
      %1052 = stablehlo.multiply %980, %980 : tensor<256xf32>
      %1053 = stablehlo.add %1051, %1052 : tensor<256xf32>
      %1054 = stablehlo.multiply %1030, %1030 : tensor<256xf32>
      %1055 = stablehlo.add %1053, %1054 : tensor<256xf32>
      %1056 = stablehlo.subtract %1032, %1055 : tensor<256xf32>
      %1057 = stablehlo.sqrt %1056 : tensor<256xf32>
      %1058 = stablehlo.slice %95 [0:1, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1059 = stablehlo.transpose %1058, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1060 = stablehlo.reshape %1059 : (tensor<256x1xf32>) -> tensor<256xf32>
      %cst_143 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1061 = stablehlo.broadcast_in_dim %cst_143, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1062 = stablehlo.subtract %1060, %1061 : tensor<256xf32>
      %1063 = stablehlo.divide %1062, %100 : tensor<256xf32>
      %1064 = stablehlo.slice %95 [1:2, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1065 = stablehlo.transpose %1064, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1066 = stablehlo.reshape %1065 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1067 = stablehlo.multiply %105, %1063 : tensor<256xf32>
      %cst_144 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1068 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1069 = stablehlo.add %1068, %1067 : tensor<256xf32>
      %1070 = stablehlo.subtract %1066, %1069 : tensor<256xf32>
      %1071 = stablehlo.divide %1070, %162 : tensor<256xf32>
      %1072 = stablehlo.slice %95 [2:3, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1073 = stablehlo.transpose %1072, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1074 = stablehlo.reshape %1073 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1075 = stablehlo.multiply %110, %1063 : tensor<256xf32>
      %cst_145 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1076 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1077 = stablehlo.add %1076, %1075 : tensor<256xf32>
      %1078 = stablehlo.multiply %169, %1071 : tensor<256xf32>
      %1079 = stablehlo.add %1077, %1078 : tensor<256xf32>
      %1080 = stablehlo.subtract %1074, %1079 : tensor<256xf32>
      %1081 = stablehlo.divide %1080, %241 : tensor<256xf32>
      %1082 = stablehlo.slice %95 [3:4, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1083 = stablehlo.transpose %1082, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1084 = stablehlo.reshape %1083 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1085 = stablehlo.multiply %115, %1063 : tensor<256xf32>
      %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1086 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1087 = stablehlo.add %1086, %1085 : tensor<256xf32>
      %1088 = stablehlo.multiply %176, %1071 : tensor<256xf32>
      %1089 = stablehlo.add %1087, %1088 : tensor<256xf32>
      %1090 = stablehlo.multiply %250, %1081 : tensor<256xf32>
      %1091 = stablehlo.add %1089, %1090 : tensor<256xf32>
      %1092 = stablehlo.subtract %1084, %1091 : tensor<256xf32>
      %1093 = stablehlo.divide %1092, %333 : tensor<256xf32>
      %1094 = stablehlo.slice %95 [4:5, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1095 = stablehlo.transpose %1094, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1096 = stablehlo.reshape %1095 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1097 = stablehlo.multiply %120, %1063 : tensor<256xf32>
      %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1098 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1099 = stablehlo.add %1098, %1097 : tensor<256xf32>
      %1100 = stablehlo.multiply %183, %1071 : tensor<256xf32>
      %1101 = stablehlo.add %1099, %1100 : tensor<256xf32>
      %1102 = stablehlo.multiply %259, %1081 : tensor<256xf32>
      %1103 = stablehlo.add %1101, %1102 : tensor<256xf32>
      %1104 = stablehlo.multiply %344, %1093 : tensor<256xf32>
      %1105 = stablehlo.add %1103, %1104 : tensor<256xf32>
      %1106 = stablehlo.subtract %1096, %1105 : tensor<256xf32>
      %1107 = stablehlo.divide %1106, %434 : tensor<256xf32>
      %1108 = stablehlo.slice %95 [5:6, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1109 = stablehlo.transpose %1108, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1110 = stablehlo.reshape %1109 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1111 = stablehlo.multiply %125, %1063 : tensor<256xf32>
      %cst_148 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1112 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1113 = stablehlo.add %1112, %1111 : tensor<256xf32>
      %1114 = stablehlo.multiply %190, %1071 : tensor<256xf32>
      %1115 = stablehlo.add %1113, %1114 : tensor<256xf32>
      %1116 = stablehlo.multiply %268, %1081 : tensor<256xf32>
      %1117 = stablehlo.add %1115, %1116 : tensor<256xf32>
      %1118 = stablehlo.multiply %355, %1093 : tensor<256xf32>
      %1119 = stablehlo.add %1117, %1118 : tensor<256xf32>
      %1120 = stablehlo.multiply %447, %1107 : tensor<256xf32>
      %1121 = stablehlo.add %1119, %1120 : tensor<256xf32>
      %1122 = stablehlo.subtract %1110, %1121 : tensor<256xf32>
      %1123 = stablehlo.divide %1122, %540 : tensor<256xf32>
      %1124 = stablehlo.slice %95 [6:7, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1125 = stablehlo.transpose %1124, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1126 = stablehlo.reshape %1125 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1127 = stablehlo.multiply %130, %1063 : tensor<256xf32>
      %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1128 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1129 = stablehlo.add %1128, %1127 : tensor<256xf32>
      %1130 = stablehlo.multiply %197, %1071 : tensor<256xf32>
      %1131 = stablehlo.add %1129, %1130 : tensor<256xf32>
      %1132 = stablehlo.multiply %277, %1081 : tensor<256xf32>
      %1133 = stablehlo.add %1131, %1132 : tensor<256xf32>
      %1134 = stablehlo.multiply %366, %1093 : tensor<256xf32>
      %1135 = stablehlo.add %1133, %1134 : tensor<256xf32>
      %1136 = stablehlo.multiply %460, %1107 : tensor<256xf32>
      %1137 = stablehlo.add %1135, %1136 : tensor<256xf32>
      %1138 = stablehlo.multiply %555, %1123 : tensor<256xf32>
      %1139 = stablehlo.add %1137, %1138 : tensor<256xf32>
      %1140 = stablehlo.subtract %1126, %1139 : tensor<256xf32>
      %1141 = stablehlo.divide %1140, %647 : tensor<256xf32>
      %1142 = stablehlo.slice %95 [7:8, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1143 = stablehlo.transpose %1142, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1144 = stablehlo.reshape %1143 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1145 = stablehlo.multiply %135, %1063 : tensor<256xf32>
      %cst_150 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1146 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1147 = stablehlo.add %1146, %1145 : tensor<256xf32>
      %1148 = stablehlo.multiply %204, %1071 : tensor<256xf32>
      %1149 = stablehlo.add %1147, %1148 : tensor<256xf32>
      %1150 = stablehlo.multiply %286, %1081 : tensor<256xf32>
      %1151 = stablehlo.add %1149, %1150 : tensor<256xf32>
      %1152 = stablehlo.multiply %377, %1093 : tensor<256xf32>
      %1153 = stablehlo.add %1151, %1152 : tensor<256xf32>
      %1154 = stablehlo.multiply %473, %1107 : tensor<256xf32>
      %1155 = stablehlo.add %1153, %1154 : tensor<256xf32>
      %1156 = stablehlo.multiply %570, %1123 : tensor<256xf32>
      %1157 = stablehlo.add %1155, %1156 : tensor<256xf32>
      %1158 = stablehlo.multiply %664, %1141 : tensor<256xf32>
      %1159 = stablehlo.add %1157, %1158 : tensor<256xf32>
      %1160 = stablehlo.subtract %1144, %1159 : tensor<256xf32>
      %1161 = stablehlo.divide %1160, %751 : tensor<256xf32>
      %1162 = stablehlo.slice %95 [8:9, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1163 = stablehlo.transpose %1162, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1164 = stablehlo.reshape %1163 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1165 = stablehlo.multiply %140, %1063 : tensor<256xf32>
      %cst_151 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1166 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1167 = stablehlo.add %1166, %1165 : tensor<256xf32>
      %1168 = stablehlo.multiply %211, %1071 : tensor<256xf32>
      %1169 = stablehlo.add %1167, %1168 : tensor<256xf32>
      %1170 = stablehlo.multiply %295, %1081 : tensor<256xf32>
      %1171 = stablehlo.add %1169, %1170 : tensor<256xf32>
      %1172 = stablehlo.multiply %388, %1093 : tensor<256xf32>
      %1173 = stablehlo.add %1171, %1172 : tensor<256xf32>
      %1174 = stablehlo.multiply %486, %1107 : tensor<256xf32>
      %1175 = stablehlo.add %1173, %1174 : tensor<256xf32>
      %1176 = stablehlo.multiply %585, %1123 : tensor<256xf32>
      %1177 = stablehlo.add %1175, %1176 : tensor<256xf32>
      %1178 = stablehlo.multiply %681, %1141 : tensor<256xf32>
      %1179 = stablehlo.add %1177, %1178 : tensor<256xf32>
      %1180 = stablehlo.multiply %770, %1161 : tensor<256xf32>
      %1181 = stablehlo.add %1179, %1180 : tensor<256xf32>
      %1182 = stablehlo.subtract %1164, %1181 : tensor<256xf32>
      %1183 = stablehlo.divide %1182, %848 : tensor<256xf32>
      %1184 = stablehlo.slice %95 [9:10, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1185 = stablehlo.transpose %1184, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1186 = stablehlo.reshape %1185 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1187 = stablehlo.multiply %145, %1063 : tensor<256xf32>
      %cst_152 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1188 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1189 = stablehlo.add %1188, %1187 : tensor<256xf32>
      %1190 = stablehlo.multiply %218, %1071 : tensor<256xf32>
      %1191 = stablehlo.add %1189, %1190 : tensor<256xf32>
      %1192 = stablehlo.multiply %304, %1081 : tensor<256xf32>
      %1193 = stablehlo.add %1191, %1192 : tensor<256xf32>
      %1194 = stablehlo.multiply %399, %1093 : tensor<256xf32>
      %1195 = stablehlo.add %1193, %1194 : tensor<256xf32>
      %1196 = stablehlo.multiply %499, %1107 : tensor<256xf32>
      %1197 = stablehlo.add %1195, %1196 : tensor<256xf32>
      %1198 = stablehlo.multiply %600, %1123 : tensor<256xf32>
      %1199 = stablehlo.add %1197, %1198 : tensor<256xf32>
      %1200 = stablehlo.multiply %698, %1141 : tensor<256xf32>
      %1201 = stablehlo.add %1199, %1200 : tensor<256xf32>
      %1202 = stablehlo.multiply %789, %1161 : tensor<256xf32>
      %1203 = stablehlo.add %1201, %1202 : tensor<256xf32>
      %1204 = stablehlo.multiply %869, %1183 : tensor<256xf32>
      %1205 = stablehlo.add %1203, %1204 : tensor<256xf32>
      %1206 = stablehlo.subtract %1186, %1205 : tensor<256xf32>
      %1207 = stablehlo.divide %1206, %934 : tensor<256xf32>
      %1208 = stablehlo.slice %95 [10:11, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1209 = stablehlo.transpose %1208, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1210 = stablehlo.reshape %1209 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1211 = stablehlo.multiply %150, %1063 : tensor<256xf32>
      %cst_153 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1212 = stablehlo.broadcast_in_dim %cst_153, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1213 = stablehlo.add %1212, %1211 : tensor<256xf32>
      %1214 = stablehlo.multiply %225, %1071 : tensor<256xf32>
      %1215 = stablehlo.add %1213, %1214 : tensor<256xf32>
      %1216 = stablehlo.multiply %313, %1081 : tensor<256xf32>
      %1217 = stablehlo.add %1215, %1216 : tensor<256xf32>
      %1218 = stablehlo.multiply %410, %1093 : tensor<256xf32>
      %1219 = stablehlo.add %1217, %1218 : tensor<256xf32>
      %1220 = stablehlo.multiply %512, %1107 : tensor<256xf32>
      %1221 = stablehlo.add %1219, %1220 : tensor<256xf32>
      %1222 = stablehlo.multiply %615, %1123 : tensor<256xf32>
      %1223 = stablehlo.add %1221, %1222 : tensor<256xf32>
      %1224 = stablehlo.multiply %715, %1141 : tensor<256xf32>
      %1225 = stablehlo.add %1223, %1224 : tensor<256xf32>
      %1226 = stablehlo.multiply %808, %1161 : tensor<256xf32>
      %1227 = stablehlo.add %1225, %1226 : tensor<256xf32>
      %1228 = stablehlo.multiply %890, %1183 : tensor<256xf32>
      %1229 = stablehlo.add %1227, %1228 : tensor<256xf32>
      %1230 = stablehlo.multiply %957, %1207 : tensor<256xf32>
      %1231 = stablehlo.add %1229, %1230 : tensor<256xf32>
      %1232 = stablehlo.subtract %1210, %1231 : tensor<256xf32>
      %1233 = stablehlo.divide %1232, %1005 : tensor<256xf32>
      %1234 = stablehlo.slice %95 [11:12, 0:256] : (tensor<12x256xf32>) -> tensor<1x256xf32>
      %1235 = stablehlo.transpose %1234, dims = [1, 0] : (tensor<1x256xf32>) -> tensor<256x1xf32>
      %1236 = stablehlo.reshape %1235 : (tensor<256x1xf32>) -> tensor<256xf32>
      %1237 = stablehlo.multiply %155, %1063 : tensor<256xf32>
      %cst_154 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1238 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1239 = stablehlo.add %1238, %1237 : tensor<256xf32>
      %1240 = stablehlo.multiply %232, %1071 : tensor<256xf32>
      %1241 = stablehlo.add %1239, %1240 : tensor<256xf32>
      %1242 = stablehlo.multiply %322, %1081 : tensor<256xf32>
      %1243 = stablehlo.add %1241, %1242 : tensor<256xf32>
      %1244 = stablehlo.multiply %421, %1093 : tensor<256xf32>
      %1245 = stablehlo.add %1243, %1244 : tensor<256xf32>
      %1246 = stablehlo.multiply %525, %1107 : tensor<256xf32>
      %1247 = stablehlo.add %1245, %1246 : tensor<256xf32>
      %1248 = stablehlo.multiply %630, %1123 : tensor<256xf32>
      %1249 = stablehlo.add %1247, %1248 : tensor<256xf32>
      %1250 = stablehlo.multiply %732, %1141 : tensor<256xf32>
      %1251 = stablehlo.add %1249, %1250 : tensor<256xf32>
      %1252 = stablehlo.multiply %827, %1161 : tensor<256xf32>
      %1253 = stablehlo.add %1251, %1252 : tensor<256xf32>
      %1254 = stablehlo.multiply %911, %1183 : tensor<256xf32>
      %1255 = stablehlo.add %1253, %1254 : tensor<256xf32>
      %1256 = stablehlo.multiply %980, %1207 : tensor<256xf32>
      %1257 = stablehlo.add %1255, %1256 : tensor<256xf32>
      %1258 = stablehlo.multiply %1030, %1233 : tensor<256xf32>
      %1259 = stablehlo.add %1257, %1258 : tensor<256xf32>
      %1260 = stablehlo.subtract %1236, %1259 : tensor<256xf32>
      %1261 = stablehlo.divide %1260, %1057 : tensor<256xf32>
      %cst_155 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1262 = stablehlo.broadcast_in_dim %cst_155, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1263 = stablehlo.subtract %1261, %1262 : tensor<256xf32>
      %1264 = stablehlo.divide %1263, %1057 : tensor<256xf32>
      %1265 = stablehlo.multiply %1030, %1264 : tensor<256xf32>
      %cst_156 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1266 = stablehlo.broadcast_in_dim %cst_156, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1267 = stablehlo.add %1266, %1265 : tensor<256xf32>
      %1268 = stablehlo.subtract %1233, %1267 : tensor<256xf32>
      %1269 = stablehlo.divide %1268, %1005 : tensor<256xf32>
      %1270 = stablehlo.multiply %957, %1269 : tensor<256xf32>
      %cst_157 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1271 = stablehlo.broadcast_in_dim %cst_157, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1272 = stablehlo.add %1271, %1270 : tensor<256xf32>
      %1273 = stablehlo.multiply %980, %1264 : tensor<256xf32>
      %1274 = stablehlo.add %1272, %1273 : tensor<256xf32>
      %1275 = stablehlo.subtract %1207, %1274 : tensor<256xf32>
      %1276 = stablehlo.divide %1275, %934 : tensor<256xf32>
      %1277 = stablehlo.multiply %869, %1276 : tensor<256xf32>
      %cst_158 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1278 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1279 = stablehlo.add %1278, %1277 : tensor<256xf32>
      %1280 = stablehlo.multiply %890, %1269 : tensor<256xf32>
      %1281 = stablehlo.add %1279, %1280 : tensor<256xf32>
      %1282 = stablehlo.multiply %911, %1264 : tensor<256xf32>
      %1283 = stablehlo.add %1281, %1282 : tensor<256xf32>
      %1284 = stablehlo.subtract %1183, %1283 : tensor<256xf32>
      %1285 = stablehlo.divide %1284, %848 : tensor<256xf32>
      %1286 = stablehlo.multiply %770, %1285 : tensor<256xf32>
      %cst_159 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1287 = stablehlo.broadcast_in_dim %cst_159, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1288 = stablehlo.add %1287, %1286 : tensor<256xf32>
      %1289 = stablehlo.multiply %789, %1276 : tensor<256xf32>
      %1290 = stablehlo.add %1288, %1289 : tensor<256xf32>
      %1291 = stablehlo.multiply %808, %1269 : tensor<256xf32>
      %1292 = stablehlo.add %1290, %1291 : tensor<256xf32>
      %1293 = stablehlo.multiply %827, %1264 : tensor<256xf32>
      %1294 = stablehlo.add %1292, %1293 : tensor<256xf32>
      %1295 = stablehlo.subtract %1161, %1294 : tensor<256xf32>
      %1296 = stablehlo.divide %1295, %751 : tensor<256xf32>
      %1297 = stablehlo.multiply %664, %1296 : tensor<256xf32>
      %cst_160 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1298 = stablehlo.broadcast_in_dim %cst_160, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1299 = stablehlo.add %1298, %1297 : tensor<256xf32>
      %1300 = stablehlo.multiply %681, %1285 : tensor<256xf32>
      %1301 = stablehlo.add %1299, %1300 : tensor<256xf32>
      %1302 = stablehlo.multiply %698, %1276 : tensor<256xf32>
      %1303 = stablehlo.add %1301, %1302 : tensor<256xf32>
      %1304 = stablehlo.multiply %715, %1269 : tensor<256xf32>
      %1305 = stablehlo.add %1303, %1304 : tensor<256xf32>
      %1306 = stablehlo.multiply %732, %1264 : tensor<256xf32>
      %1307 = stablehlo.add %1305, %1306 : tensor<256xf32>
      %1308 = stablehlo.subtract %1141, %1307 : tensor<256xf32>
      %1309 = stablehlo.divide %1308, %647 : tensor<256xf32>
      %1310 = stablehlo.multiply %555, %1309 : tensor<256xf32>
      %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1311 = stablehlo.broadcast_in_dim %cst_161, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1312 = stablehlo.add %1311, %1310 : tensor<256xf32>
      %1313 = stablehlo.multiply %570, %1296 : tensor<256xf32>
      %1314 = stablehlo.add %1312, %1313 : tensor<256xf32>
      %1315 = stablehlo.multiply %585, %1285 : tensor<256xf32>
      %1316 = stablehlo.add %1314, %1315 : tensor<256xf32>
      %1317 = stablehlo.multiply %600, %1276 : tensor<256xf32>
      %1318 = stablehlo.add %1316, %1317 : tensor<256xf32>
      %1319 = stablehlo.multiply %615, %1269 : tensor<256xf32>
      %1320 = stablehlo.add %1318, %1319 : tensor<256xf32>
      %1321 = stablehlo.multiply %630, %1264 : tensor<256xf32>
      %1322 = stablehlo.add %1320, %1321 : tensor<256xf32>
      %1323 = stablehlo.subtract %1123, %1322 : tensor<256xf32>
      %1324 = stablehlo.divide %1323, %540 : tensor<256xf32>
      %1325 = stablehlo.multiply %447, %1324 : tensor<256xf32>
      %cst_162 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1326 = stablehlo.broadcast_in_dim %cst_162, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1327 = stablehlo.add %1326, %1325 : tensor<256xf32>
      %1328 = stablehlo.multiply %460, %1309 : tensor<256xf32>
      %1329 = stablehlo.add %1327, %1328 : tensor<256xf32>
      %1330 = stablehlo.multiply %473, %1296 : tensor<256xf32>
      %1331 = stablehlo.add %1329, %1330 : tensor<256xf32>
      %1332 = stablehlo.multiply %486, %1285 : tensor<256xf32>
      %1333 = stablehlo.add %1331, %1332 : tensor<256xf32>
      %1334 = stablehlo.multiply %499, %1276 : tensor<256xf32>
      %1335 = stablehlo.add %1333, %1334 : tensor<256xf32>
      %1336 = stablehlo.multiply %512, %1269 : tensor<256xf32>
      %1337 = stablehlo.add %1335, %1336 : tensor<256xf32>
      %1338 = stablehlo.multiply %525, %1264 : tensor<256xf32>
      %1339 = stablehlo.add %1337, %1338 : tensor<256xf32>
      %1340 = stablehlo.subtract %1107, %1339 : tensor<256xf32>
      %1341 = stablehlo.divide %1340, %434 : tensor<256xf32>
      %1342 = stablehlo.multiply %344, %1341 : tensor<256xf32>
      %cst_163 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1343 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1344 = stablehlo.add %1343, %1342 : tensor<256xf32>
      %1345 = stablehlo.multiply %355, %1324 : tensor<256xf32>
      %1346 = stablehlo.add %1344, %1345 : tensor<256xf32>
      %1347 = stablehlo.multiply %366, %1309 : tensor<256xf32>
      %1348 = stablehlo.add %1346, %1347 : tensor<256xf32>
      %1349 = stablehlo.multiply %377, %1296 : tensor<256xf32>
      %1350 = stablehlo.add %1348, %1349 : tensor<256xf32>
      %1351 = stablehlo.multiply %388, %1285 : tensor<256xf32>
      %1352 = stablehlo.add %1350, %1351 : tensor<256xf32>
      %1353 = stablehlo.multiply %399, %1276 : tensor<256xf32>
      %1354 = stablehlo.add %1352, %1353 : tensor<256xf32>
      %1355 = stablehlo.multiply %410, %1269 : tensor<256xf32>
      %1356 = stablehlo.add %1354, %1355 : tensor<256xf32>
      %1357 = stablehlo.multiply %421, %1264 : tensor<256xf32>
      %1358 = stablehlo.add %1356, %1357 : tensor<256xf32>
      %1359 = stablehlo.subtract %1093, %1358 : tensor<256xf32>
      %1360 = stablehlo.divide %1359, %333 : tensor<256xf32>
      %1361 = stablehlo.multiply %250, %1360 : tensor<256xf32>
      %cst_164 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1362 = stablehlo.broadcast_in_dim %cst_164, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1363 = stablehlo.add %1362, %1361 : tensor<256xf32>
      %1364 = stablehlo.multiply %259, %1341 : tensor<256xf32>
      %1365 = stablehlo.add %1363, %1364 : tensor<256xf32>
      %1366 = stablehlo.multiply %268, %1324 : tensor<256xf32>
      %1367 = stablehlo.add %1365, %1366 : tensor<256xf32>
      %1368 = stablehlo.multiply %277, %1309 : tensor<256xf32>
      %1369 = stablehlo.add %1367, %1368 : tensor<256xf32>
      %1370 = stablehlo.multiply %286, %1296 : tensor<256xf32>
      %1371 = stablehlo.add %1369, %1370 : tensor<256xf32>
      %1372 = stablehlo.multiply %295, %1285 : tensor<256xf32>
      %1373 = stablehlo.add %1371, %1372 : tensor<256xf32>
      %1374 = stablehlo.multiply %304, %1276 : tensor<256xf32>
      %1375 = stablehlo.add %1373, %1374 : tensor<256xf32>
      %1376 = stablehlo.multiply %313, %1269 : tensor<256xf32>
      %1377 = stablehlo.add %1375, %1376 : tensor<256xf32>
      %1378 = stablehlo.multiply %322, %1264 : tensor<256xf32>
      %1379 = stablehlo.add %1377, %1378 : tensor<256xf32>
      %1380 = stablehlo.subtract %1081, %1379 : tensor<256xf32>
      %1381 = stablehlo.divide %1380, %241 : tensor<256xf32>
      %1382 = stablehlo.multiply %169, %1381 : tensor<256xf32>
      %cst_165 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1383 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1384 = stablehlo.add %1383, %1382 : tensor<256xf32>
      %1385 = stablehlo.multiply %176, %1360 : tensor<256xf32>
      %1386 = stablehlo.add %1384, %1385 : tensor<256xf32>
      %1387 = stablehlo.multiply %183, %1341 : tensor<256xf32>
      %1388 = stablehlo.add %1386, %1387 : tensor<256xf32>
      %1389 = stablehlo.multiply %190, %1324 : tensor<256xf32>
      %1390 = stablehlo.add %1388, %1389 : tensor<256xf32>
      %1391 = stablehlo.multiply %197, %1309 : tensor<256xf32>
      %1392 = stablehlo.add %1390, %1391 : tensor<256xf32>
      %1393 = stablehlo.multiply %204, %1296 : tensor<256xf32>
      %1394 = stablehlo.add %1392, %1393 : tensor<256xf32>
      %1395 = stablehlo.multiply %211, %1285 : tensor<256xf32>
      %1396 = stablehlo.add %1394, %1395 : tensor<256xf32>
      %1397 = stablehlo.multiply %218, %1276 : tensor<256xf32>
      %1398 = stablehlo.add %1396, %1397 : tensor<256xf32>
      %1399 = stablehlo.multiply %225, %1269 : tensor<256xf32>
      %1400 = stablehlo.add %1398, %1399 : tensor<256xf32>
      %1401 = stablehlo.multiply %232, %1264 : tensor<256xf32>
      %1402 = stablehlo.add %1400, %1401 : tensor<256xf32>
      %1403 = stablehlo.subtract %1071, %1402 : tensor<256xf32>
      %1404 = stablehlo.divide %1403, %162 : tensor<256xf32>
      %1405 = stablehlo.multiply %105, %1404 : tensor<256xf32>
      %cst_166 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1406 = stablehlo.broadcast_in_dim %cst_166, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1407 = stablehlo.add %1406, %1405 : tensor<256xf32>
      %1408 = stablehlo.multiply %110, %1381 : tensor<256xf32>
      %1409 = stablehlo.add %1407, %1408 : tensor<256xf32>
      %1410 = stablehlo.multiply %115, %1360 : tensor<256xf32>
      %1411 = stablehlo.add %1409, %1410 : tensor<256xf32>
      %1412 = stablehlo.multiply %120, %1341 : tensor<256xf32>
      %1413 = stablehlo.add %1411, %1412 : tensor<256xf32>
      %1414 = stablehlo.multiply %125, %1324 : tensor<256xf32>
      %1415 = stablehlo.add %1413, %1414 : tensor<256xf32>
      %1416 = stablehlo.multiply %130, %1309 : tensor<256xf32>
      %1417 = stablehlo.add %1415, %1416 : tensor<256xf32>
      %1418 = stablehlo.multiply %135, %1296 : tensor<256xf32>
      %1419 = stablehlo.add %1417, %1418 : tensor<256xf32>
      %1420 = stablehlo.multiply %140, %1285 : tensor<256xf32>
      %1421 = stablehlo.add %1419, %1420 : tensor<256xf32>
      %1422 = stablehlo.multiply %145, %1276 : tensor<256xf32>
      %1423 = stablehlo.add %1421, %1422 : tensor<256xf32>
      %1424 = stablehlo.multiply %150, %1269 : tensor<256xf32>
      %1425 = stablehlo.add %1423, %1424 : tensor<256xf32>
      %1426 = stablehlo.multiply %155, %1264 : tensor<256xf32>
      %1427 = stablehlo.add %1425, %1426 : tensor<256xf32>
      %1428 = stablehlo.subtract %1063, %1427 : tensor<256xf32>
      %1429 = stablehlo.divide %1428, %100 : tensor<256xf32>
      %1430 = stablehlo.broadcast_in_dim %1429, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1431 = stablehlo.broadcast_in_dim %1404, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1432 = stablehlo.broadcast_in_dim %1381, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1433 = stablehlo.broadcast_in_dim %1360, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1434 = stablehlo.broadcast_in_dim %1341, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1435 = stablehlo.broadcast_in_dim %1324, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1436 = stablehlo.broadcast_in_dim %1309, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1437 = stablehlo.broadcast_in_dim %1296, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1438 = stablehlo.broadcast_in_dim %1285, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1439 = stablehlo.broadcast_in_dim %1276, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1440 = stablehlo.broadcast_in_dim %1269, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1441 = stablehlo.broadcast_in_dim %1264, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
      %1442 = stablehlo.concatenate %1430, %1431, %1432, %1433, %1434, %1435, %1436, %1437, %1438, %1439, %1440, %1441, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<12x256xf32>
      %c_167 = stablehlo.constant dense<false> : tensor<i1>
      %1443 = func.call @_where_2(%c_167, %18, %1442) : (tensor<i1>, tensor<256x12xf32>, tensor<12x256xf32>) -> tensor<256x12xf32>
      %1444 = stablehlo.dot_general %iterArg_31, %1443, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<8x12xf32>, tensor<256x12xf32>) -> tensor<8x256xf32>
      %1445 = stablehlo.transpose %1444, dims = [1, 0] : (tensor<8x256xf32>) -> tensor<256x8xf32>
      %1446 = stablehlo.add %1445, %15 : tensor<256x8xf32>
      %1447 = stablehlo.exponential %1446 : tensor<256x8xf32>
      %cst_168 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1448 = stablehlo.broadcast_in_dim %cst_168, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %1449 = stablehlo.maximum %1447, %1448 : tensor<256x8xf32>
      %cst_169 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1450 = stablehlo.broadcast_in_dim %cst_169, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %1451 = stablehlo.compare NE, %14, %1450, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
      %1452 = stablehlo.compare NE, %14, %14, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
      %1453 = stablehlo.or %1451, %1452 : tensor<256x8xi1>
      %1454 = stablehlo.log %14 : tensor<256x8xf32>
      %1455 = stablehlo.multiply %14, %1454 : tensor<256x8xf32>
      %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1456 = stablehlo.broadcast_in_dim %cst_170, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %1457 = func.call @_where_3(%1453, %1455, %1456) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
      %cst_171 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1458 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %1459 = stablehlo.compare NE, %14, %1458, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
      %1460 = stablehlo.compare NE, %1449, %1449, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
      %1461 = stablehlo.or %1459, %1460 : tensor<256x8xi1>
      %1462 = stablehlo.log %1449 : tensor<256x8xf32>
      %1463 = stablehlo.multiply %14, %1462 : tensor<256x8xf32>
      %cst_172 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1464 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %1465 = func.call @_where_3(%1461, %1463, %1464) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
      %1466 = stablehlo.subtract %1457, %1465 : tensor<256x8xf32>
      %1467 = stablehlo.subtract %14, %1449 : tensor<256x8xf32>
      %1468 = stablehlo.subtract %1466, %1467 : tensor<256x8xf32>
      %cst_173 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1469 = stablehlo.broadcast_in_dim %cst_173, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %1470 = stablehlo.multiply %1469, %1468 : tensor<256x8xf32>
      %cst_174 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1471 = stablehlo.broadcast_in_dim %cst_174, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1472 = stablehlo.compare GT, %17, %1471, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %cst_175 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
      %1473 = func.call @_where_4(%1472, %17, %cst_175) : (tensor<256xi1>, tensor<256xf32>, tensor<f64>) -> tensor<256xf32>
      %cst_176 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1474 = stablehlo.broadcast_in_dim %cst_176, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %1475 = stablehlo.compare NE, %14, %1474, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
      %1476 = stablehlo.compare NE, %14, %14, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
      %1477 = stablehlo.or %1475, %1476 : tensor<256x8xi1>
      %1478 = stablehlo.log %14 : tensor<256x8xf32>
      %1479 = stablehlo.multiply %14, %1478 : tensor<256x8xf32>
      %cst_177 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1480 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %1481 = func.call @_where_3(%1477, %1479, %1480) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
      %cst_178 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1482 = stablehlo.broadcast_in_dim %cst_178, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %1483 = stablehlo.compare NE, %14, %1482, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
      %1484 = stablehlo.compare NE, %1449, %1449, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
      %1485 = stablehlo.or %1483, %1484 : tensor<256x8xi1>
      %1486 = stablehlo.log %1449 : tensor<256x8xf32>
      %1487 = stablehlo.multiply %14, %1486 : tensor<256x8xf32>
      %cst_179 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1488 = stablehlo.broadcast_in_dim %cst_179, dims = [] : (tensor<f32>) -> tensor<8xf32>
      %1489 = func.call @_where_3(%1485, %1487, %1488) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
      %1490 = stablehlo.subtract %1481, %1489 : tensor<256x8xf32>
      %cst_180 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %1491 = stablehlo.broadcast_in_dim %cst_180, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1492 = stablehlo.divide %1491, %1473 : tensor<256xf32>
      %1493 = stablehlo.broadcast_in_dim %1492, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
      %1494 = stablehlo.broadcast_in_dim %1493, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
      %1495 = stablehlo.add %14, %1494 : tensor<256x8xf32>
      %1496 = stablehlo.broadcast_in_dim %1473, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
      %1497 = stablehlo.broadcast_in_dim %1496, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
      %1498 = stablehlo.multiply %1497, %14 : tensor<256x8xf32>
      %1499 = stablehlo.log_plus_one %1498 : tensor<256x8xf32>
      %1500 = stablehlo.broadcast_in_dim %1473, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
      %1501 = stablehlo.broadcast_in_dim %1500, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
      %1502 = stablehlo.multiply %1501, %1449 : tensor<256x8xf32>
      %1503 = stablehlo.log_plus_one %1502 : tensor<256x8xf32>
      %1504 = stablehlo.subtract %1499, %1503 : tensor<256x8xf32>
      %1505 = stablehlo.multiply %1495, %1504 : tensor<256x8xf32>
      %1506 = stablehlo.subtract %1490, %1505 : tensor<256x8xf32>
      %cst_181 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
      %1507 = stablehlo.broadcast_in_dim %cst_181, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
      %1508 = stablehlo.multiply %1507, %1506 : tensor<256x8xf32>
      %cst_182 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1509 = stablehlo.broadcast_in_dim %cst_182, dims = [] : (tensor<f32>) -> tensor<256xf32>
      %1510 = stablehlo.compare GT, %17, %1509, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
      %1511 = func.call @_where_5(%1510, %1508, %1470) : (tensor<256xi1>, tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xf32>
      %1512 = stablehlo.multiply %16, %1511 : tensor<256x8xf32>
      %cst_183 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %1513 = stablehlo.reduce(%1512 init: %cst_183) applies stablehlo.add across dimensions = [1] : (tensor<256x8xf32>, tensor<f32>) -> tensor<256xf32>
      %c_184 = stablehlo.constant dense<0> : tensor<i64>
      %1514 = stablehlo.broadcast_in_dim %c_184, dims = [] : (tensor<i64>) -> tensor<256xi64>
      %c_185 = stablehlo.constant dense<false> : tensor<i1>
      %1515 = stablehlo.broadcast_in_dim %c_185, dims = [] : (tensor<i1>) -> tensor<256xi1>
      %1516:17 = stablehlo.while(%iterArg_187 = %iterArg_31, %iterArg_188 = %15, %iterArg_189 = %17, %iterArg_190 = %14, %iterArg_191 = %16, %iterArg_192 = %iterArg_40, %iterArg_193 = %iterArg_41, %iterArg_194 = %iterArg_42, %iterArg_195 = %iterArg_43, %iterArg_196 = %iterArg_44, %iterArg_197 = %iterArg_45, %iterArg_198 = %iterArg_46, %iterArg_199 = %iterArg_47, %iterArg_200 = %1443, %iterArg_201 = %1513, %iterArg_202 = %1514, %iterArg_203 = %1515) : tensor<8x12xf32>, tensor<256x8xf32>, tensor<256xf32>, tensor<256x8xf32>, tensor<256x8xf32>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<256x12xf32>, tensor<256xf32>, tensor<256xi64>, tensor<256xi1>
      cond {
        %c_204 = stablehlo.constant dense<50> : tensor<i64>
        %1522 = stablehlo.broadcast_in_dim %c_204, dims = [] : (tensor<i64>) -> tensor<256xi64>
        %1523 = stablehlo.compare LT, %iterArg_202, %1522, SIGNED : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
        %1524 = stablehlo.not %iterArg_203 : tensor<256xi1>
        %c_205 = stablehlo.constant dense<false> : tensor<i1>
        %1525 = stablehlo.broadcast_in_dim %c_205, dims = [] : (tensor<i1>) -> tensor<256xi1>
        %1526 = stablehlo.compare NE, %1523, %1525, UNSIGNED : (tensor<256xi1>, tensor<256xi1>) -> tensor<256xi1>
        %1527 = stablehlo.convert %1526 : tensor<256xi1>
        %1528 = stablehlo.and %1527, %1524 : tensor<256xi1>
        %c_206 = stablehlo.constant dense<false> : tensor<i1>
        %1529 = stablehlo.reduce(%1528 init: %c_206) applies stablehlo.or across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
        stablehlo.return %1529 : tensor<i1>
      } do {
        %1522 = stablehlo.dot_general %iterArg_187, %iterArg_200, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<8x12xf32>, tensor<256x12xf32>) -> tensor<8x256xf32>
        %1523 = stablehlo.transpose %1522, dims = [1, 0] : (tensor<8x256xf32>) -> tensor<256x8xf32>
        %1524 = stablehlo.add %1523, %iterArg_188 : tensor<256x8xf32>
        %cst_204 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %1525 = stablehlo.broadcast_in_dim %cst_204, dims = [] : (tensor<f32>) -> tensor<8xf32>
        %1526 = stablehlo.exponential %1524 : tensor<256x8xf32>
        %1527 = stablehlo.broadcast_in_dim %1525, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
        %1528 = stablehlo.broadcast_in_dim %1527, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<256x8xf32>
        %1529 = stablehlo.multiply %1528, %1526 : tensor<256x8xf32>
        %cst_205 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1530 = stablehlo.broadcast_in_dim %cst_205, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %1531 = stablehlo.maximum %1526, %1530 : tensor<256x8xf32>
        %cst_206 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1532 = stablehlo.broadcast_in_dim %cst_206, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %1533 = stablehlo.compare GT, %1526, %1532, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %cst_207 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1534 = stablehlo.broadcast_in_dim %cst_207, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %1535 = stablehlo.compare EQ, %1526, %1534, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %cst_208 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %1536 = stablehlo.broadcast_in_dim %cst_208, dims = [] : (tensor<f32>) -> tensor<8xf32>
        %cst_209 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1537 = stablehlo.broadcast_in_dim %cst_209, dims = [] : (tensor<f32>) -> tensor<8xf32>
        %cst_210 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
        %1538 = stablehlo.broadcast_in_dim %cst_210, dims = [] : (tensor<f32>) -> tensor<8xf32>
        %1539 = stablehlo.broadcast_in_dim %1537, dims = [1] : (tensor<8xf32>) -> tensor<256x8xf32>
        %1540 = stablehlo.broadcast_in_dim %1538, dims = [1] : (tensor<8xf32>) -> tensor<256x8xf32>
        %1541 = stablehlo.select %1535, %1540, %1539 : tensor<256x8xi1>, tensor<256x8xf32>
        %1542 = stablehlo.broadcast_in_dim %1536, dims = [1] : (tensor<8xf32>) -> tensor<256x8xf32>
        %1543 = stablehlo.select %1533, %1542, %1541 : tensor<256x8xi1>, tensor<256x8xf32>
        %1544 = stablehlo.multiply %1529, %1543 : tensor<256x8xf32>
        %1545 = stablehlo.exponential %1524 : tensor<256x8xf32>
        %1546 = stablehlo.broadcast_in_dim %1525, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
        %1547 = stablehlo.broadcast_in_dim %1546, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<256x8xf32>
        %1548 = stablehlo.multiply %1547, %1545 : tensor<256x8xf32>
        %1549 = stablehlo.multiply %1531, %1531 : tensor<256x8xf32>
        %cst_211 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
        %1550 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %1551 = stablehlo.multiply %1550, %1531 : tensor<256x8xf32>
        %1552 = stablehlo.multiply %1544, %1551 : tensor<256x8xf32>
        %1553 = stablehlo.broadcast_in_dim %iterArg_189, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
        %1554 = stablehlo.broadcast_in_dim %1553, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
        %1555 = stablehlo.multiply %1554, %1549 : tensor<256x8xf32>
        %1556 = stablehlo.broadcast_in_dim %iterArg_189, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
        %1557 = stablehlo.broadcast_in_dim %1556, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
        %1558 = stablehlo.multiply %1557, %1552 : tensor<256x8xf32>
        %1559 = stablehlo.add %1531, %1555 : tensor<256x8xf32>
        %1560 = stablehlo.add %1544, %1558 : tensor<256x8xf32>
        %1561 = stablehlo.divide %1545, %1559 : tensor<256x8xf32>
        %1562 = stablehlo.divide %1548, %1559 : tensor<256x8xf32>
        %1563 = stablehlo.negate %1560 : tensor<256x8xf32>
        %1564 = stablehlo.multiply %1563, %1545 : tensor<256x8xf32>
        %1565 = stablehlo.multiply %1559, %1559 : tensor<256x8xf32>
        %cst_212 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %1566 = stablehlo.broadcast_in_dim %cst_212, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %1567 = stablehlo.divide %1566, %1565 : tensor<256x8xf32>
        %1568 = stablehlo.multiply %1564, %1567 : tensor<256x8xf32>
        %1569 = stablehlo.add %1562, %1568 : tensor<256x8xf32>
        %1570 = stablehlo.exponential %1524 : tensor<256x8xf32>
        %1571 = stablehlo.subtract %iterArg_190, %1531 : tensor<256x8xf32>
        %1572 = stablehlo.multiply %iterArg_191, %1571 : tensor<256x8xf32>
        %1573 = stablehlo.multiply %iterArg_191, %1570 : tensor<256x8xf32>
        %1574 = stablehlo.multiply %1573, %1561 : tensor<256x8xf32>
        %1575 = stablehlo.multiply %1572, %1569 : tensor<256x8xf32>
        %1576 = stablehlo.subtract %1574, %1575 : tensor<256x8xf32>
        %cst_213 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1577 = stablehlo.broadcast_in_dim %cst_213, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %1578 = stablehlo.compare GT, %1576, %1577, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %1579 = func.call @_where_6(%1578, %1576, %1574) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xf32>
        %1580 = stablehlo.multiply %1572, %1561 : tensor<256x8xf32>
        %c_214 = stablehlo.constant dense<12> : tensor<i64>
        %1581 = stablehlo.broadcast_in_dim %c_214, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1582 = stablehlo.add %iterArg_192, %1581 : tensor<78xi64>
        %1583 = stablehlo.select %iterArg_193, %1582, %iterArg_192 : tensor<78xi1>, tensor<78xi64>
        %1584 = stablehlo.convert %1583 : (tensor<78xi64>) -> tensor<78xi32>
        %1585 = stablehlo.broadcast_in_dim %1584, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1586 = "stablehlo.gather"(%iterArg_187, %1585) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x12xf32>, tensor<78x1xi32>) -> tensor<8x78xf32>
        %c_215 = stablehlo.constant dense<12> : tensor<i64>
        %1587 = stablehlo.broadcast_in_dim %c_215, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1588 = stablehlo.add %iterArg_194, %1587 : tensor<78xi64>
        %1589 = stablehlo.select %iterArg_195, %1588, %iterArg_194 : tensor<78xi1>, tensor<78xi64>
        %1590 = stablehlo.convert %1589 : (tensor<78xi64>) -> tensor<78xi32>
        %1591 = stablehlo.broadcast_in_dim %1590, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1592 = "stablehlo.gather"(%iterArg_187, %1591) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 8, 1>}> : (tensor<8x12xf32>, tensor<78x1xi32>) -> tensor<8x78xf32>
        %1593 = stablehlo.multiply %1586, %1592 : tensor<8x78xf32>
        %1594 = stablehlo.dot_general %1579, %1593, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<256x8xf32>, tensor<8x78xf32>) -> tensor<256x78xf32>
        %cst_216 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1595 = stablehlo.broadcast_in_dim %cst_216, dims = [] : (tensor<f32>) -> tensor<12x12xf32>
        %c_217 = stablehlo.constant dense<12> : tensor<i64>
        %1596 = stablehlo.broadcast_in_dim %c_217, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1597 = stablehlo.add %iterArg_192, %1596 : tensor<78xi64>
        %1598 = stablehlo.select %iterArg_196, %1597, %iterArg_192 : tensor<78xi1>, tensor<78xi64>
        %c_218 = stablehlo.constant dense<12> : tensor<i64>
        %1599 = stablehlo.broadcast_in_dim %c_218, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1600 = stablehlo.add %iterArg_194, %1599 : tensor<78xi64>
        %1601 = stablehlo.select %iterArg_197, %1600, %iterArg_194 : tensor<78xi1>, tensor<78xi64>
        %1602 = stablehlo.convert %1598 : (tensor<78xi64>) -> tensor<78xi32>
        %1603 = stablehlo.convert %1601 : (tensor<78xi64>) -> tensor<78xi32>
        %1604 = stablehlo.broadcast_in_dim %1602, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1605 = stablehlo.broadcast_in_dim %1603, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1606 = stablehlo.concatenate %1604, %1605, dim = 1 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x2xi32>
        %1607 = stablehlo.broadcast_in_dim %1595, dims = [1, 2] : (tensor<12x12xf32>) -> tensor<256x12x12xf32>
        %1608 = "stablehlo.scatter"(%1607, %1606, %1594) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
          stablehlo.return %arg7 : tensor<f32>
        }) : (tensor<256x12x12xf32>, tensor<78x2xi32>, tensor<256x78xf32>) -> tensor<256x12x12xf32>
        %c_219 = stablehlo.constant dense<12> : tensor<i64>
        %1609 = stablehlo.broadcast_in_dim %c_219, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1610 = stablehlo.add %iterArg_194, %1609 : tensor<78xi64>
        %1611 = stablehlo.select %iterArg_198, %1610, %iterArg_194 : tensor<78xi1>, tensor<78xi64>
        %c_220 = stablehlo.constant dense<12> : tensor<i64>
        %1612 = stablehlo.broadcast_in_dim %c_220, dims = [] : (tensor<i64>) -> tensor<78xi64>
        %1613 = stablehlo.add %iterArg_192, %1612 : tensor<78xi64>
        %1614 = stablehlo.select %iterArg_199, %1613, %iterArg_192 : tensor<78xi1>, tensor<78xi64>
        %1615 = stablehlo.convert %1611 : (tensor<78xi64>) -> tensor<78xi32>
        %1616 = stablehlo.convert %1614 : (tensor<78xi64>) -> tensor<78xi32>
        %1617 = stablehlo.broadcast_in_dim %1615, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1618 = stablehlo.broadcast_in_dim %1616, dims = [0] : (tensor<78xi32>) -> tensor<78x1xi32>
        %1619 = stablehlo.concatenate %1617, %1618, dim = 1 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x2xi32>
        %1620 = "stablehlo.scatter"(%1608, %1619, %1594) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 1>, unique_indices = false}> ({
        ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):
          stablehlo.return %arg7 : tensor<f32>
        }) : (tensor<256x12x12xf32>, tensor<78x2xi32>, tensor<256x78xf32>) -> tensor<256x12x12xf32>
        %1621 = stablehlo.iota dim = 0 : tensor<12x12xi64>
        %1622 = stablehlo.iota dim = 1 : tensor<12x12xi64>
        %c_221 = stablehlo.constant dense<0> : tensor<i64>
        %1623 = stablehlo.broadcast_in_dim %c_221, dims = [] : (tensor<i64>) -> tensor<12x12xi64>
        %1624 = stablehlo.add %1621, %1623 : tensor<12x12xi64>
        %1625 = stablehlo.compare EQ, %1624, %1622, SIGNED : (tensor<12x12xi64>, tensor<12x12xi64>) -> tensor<12x12xi1>
        %1626 = stablehlo.convert %1625 : (tensor<12x12xi1>) -> tensor<12x12xf32>
        %cst_222 = stablehlo.constant dense<1.000000e-10> : tensor<f32>
        %1627 = stablehlo.broadcast_in_dim %cst_222, dims = [] : (tensor<f32>) -> tensor<12x12xf32>
        %1628 = stablehlo.multiply %1627, %1626 : tensor<12x12xf32>
        %1629 = stablehlo.broadcast_in_dim %1628, dims = [1, 2] : (tensor<12x12xf32>) -> tensor<1x12x12xf32>
        %1630 = stablehlo.broadcast_in_dim %1629, dims = [0, 1, 2] : (tensor<1x12x12xf32>) -> tensor<256x12x12xf32>
        %1631 = stablehlo.add %1620, %1630 : tensor<256x12x12xf32>
        %1632 = stablehlo.transpose %iterArg_187, dims = [1, 0] : (tensor<8x12xf32>) -> tensor<12x8xf32>
        %1633 = stablehlo.dot_general %1632, %1580, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<12x8xf32>, tensor<256x8xf32>) -> tensor<12x256xf32>
        %cst_223 = stablehlo.constant dense<1.000000e-10> : tensor<f32>
        %1634 = stablehlo.broadcast_in_dim %cst_223, dims = [] : (tensor<f32>) -> tensor<256x12xf32>
        %1635 = stablehlo.multiply %1634, %iterArg_200 : tensor<256x12xf32>
        %1636 = stablehlo.transpose %1633, dims = [1, 0] : (tensor<12x256xf32>) -> tensor<256x12xf32>
        %1637 = stablehlo.subtract %1636, %1635 : tensor<256x12xf32>
        %1638 = stablehlo.slice %1631 [0:256, 0:1, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1639 = stablehlo.reshape %1638 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_224 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1640 = stablehlo.broadcast_in_dim %cst_224, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1641 = stablehlo.subtract %1639, %1640 : tensor<256xf32>
        %1642 = stablehlo.sqrt %1641 : tensor<256xf32>
        %1643 = stablehlo.slice %1631 [0:256, 1:2, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1644 = stablehlo.reshape %1643 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_225 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1645 = stablehlo.broadcast_in_dim %cst_225, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1646 = stablehlo.subtract %1644, %1645 : tensor<256xf32>
        %1647 = stablehlo.divide %1646, %1642 : tensor<256xf32>
        %1648 = stablehlo.slice %1631 [0:256, 2:3, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1649 = stablehlo.reshape %1648 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_226 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1650 = stablehlo.broadcast_in_dim %cst_226, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1651 = stablehlo.subtract %1649, %1650 : tensor<256xf32>
        %1652 = stablehlo.divide %1651, %1642 : tensor<256xf32>
        %1653 = stablehlo.slice %1631 [0:256, 3:4, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1654 = stablehlo.reshape %1653 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_227 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1655 = stablehlo.broadcast_in_dim %cst_227, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1656 = stablehlo.subtract %1654, %1655 : tensor<256xf32>
        %1657 = stablehlo.divide %1656, %1642 : tensor<256xf32>
        %1658 = stablehlo.slice %1631 [0:256, 4:5, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1659 = stablehlo.reshape %1658 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_228 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1660 = stablehlo.broadcast_in_dim %cst_228, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1661 = stablehlo.subtract %1659, %1660 : tensor<256xf32>
        %1662 = stablehlo.divide %1661, %1642 : tensor<256xf32>
        %1663 = stablehlo.slice %1631 [0:256, 5:6, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1664 = stablehlo.reshape %1663 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_229 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1665 = stablehlo.broadcast_in_dim %cst_229, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1666 = stablehlo.subtract %1664, %1665 : tensor<256xf32>
        %1667 = stablehlo.divide %1666, %1642 : tensor<256xf32>
        %1668 = stablehlo.slice %1631 [0:256, 6:7, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1669 = stablehlo.reshape %1668 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_230 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1670 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1671 = stablehlo.subtract %1669, %1670 : tensor<256xf32>
        %1672 = stablehlo.divide %1671, %1642 : tensor<256xf32>
        %1673 = stablehlo.slice %1631 [0:256, 7:8, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1674 = stablehlo.reshape %1673 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_231 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1675 = stablehlo.broadcast_in_dim %cst_231, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1676 = stablehlo.subtract %1674, %1675 : tensor<256xf32>
        %1677 = stablehlo.divide %1676, %1642 : tensor<256xf32>
        %1678 = stablehlo.slice %1631 [0:256, 8:9, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1679 = stablehlo.reshape %1678 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_232 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1680 = stablehlo.broadcast_in_dim %cst_232, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1681 = stablehlo.subtract %1679, %1680 : tensor<256xf32>
        %1682 = stablehlo.divide %1681, %1642 : tensor<256xf32>
        %1683 = stablehlo.slice %1631 [0:256, 9:10, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1684 = stablehlo.reshape %1683 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_233 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1685 = stablehlo.broadcast_in_dim %cst_233, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1686 = stablehlo.subtract %1684, %1685 : tensor<256xf32>
        %1687 = stablehlo.divide %1686, %1642 : tensor<256xf32>
        %1688 = stablehlo.slice %1631 [0:256, 10:11, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1689 = stablehlo.reshape %1688 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_234 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1690 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1691 = stablehlo.subtract %1689, %1690 : tensor<256xf32>
        %1692 = stablehlo.divide %1691, %1642 : tensor<256xf32>
        %1693 = stablehlo.slice %1631 [0:256, 11:12, 0:1] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1694 = stablehlo.reshape %1693 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %cst_235 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1695 = stablehlo.broadcast_in_dim %cst_235, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1696 = stablehlo.subtract %1694, %1695 : tensor<256xf32>
        %1697 = stablehlo.divide %1696, %1642 : tensor<256xf32>
        %1698 = stablehlo.slice %1631 [0:256, 1:2, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1699 = stablehlo.reshape %1698 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1700 = stablehlo.multiply %1647, %1647 : tensor<256xf32>
        %cst_236 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1701 = stablehlo.broadcast_in_dim %cst_236, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1702 = stablehlo.add %1701, %1700 : tensor<256xf32>
        %1703 = stablehlo.subtract %1699, %1702 : tensor<256xf32>
        %1704 = stablehlo.sqrt %1703 : tensor<256xf32>
        %1705 = stablehlo.slice %1631 [0:256, 2:3, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1706 = stablehlo.reshape %1705 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1707 = stablehlo.multiply %1652, %1647 : tensor<256xf32>
        %cst_237 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1708 = stablehlo.broadcast_in_dim %cst_237, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1709 = stablehlo.add %1708, %1707 : tensor<256xf32>
        %1710 = stablehlo.subtract %1706, %1709 : tensor<256xf32>
        %1711 = stablehlo.divide %1710, %1704 : tensor<256xf32>
        %1712 = stablehlo.slice %1631 [0:256, 3:4, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1713 = stablehlo.reshape %1712 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1714 = stablehlo.multiply %1657, %1647 : tensor<256xf32>
        %cst_238 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1715 = stablehlo.broadcast_in_dim %cst_238, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1716 = stablehlo.add %1715, %1714 : tensor<256xf32>
        %1717 = stablehlo.subtract %1713, %1716 : tensor<256xf32>
        %1718 = stablehlo.divide %1717, %1704 : tensor<256xf32>
        %1719 = stablehlo.slice %1631 [0:256, 4:5, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1720 = stablehlo.reshape %1719 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1721 = stablehlo.multiply %1662, %1647 : tensor<256xf32>
        %cst_239 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1722 = stablehlo.broadcast_in_dim %cst_239, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1723 = stablehlo.add %1722, %1721 : tensor<256xf32>
        %1724 = stablehlo.subtract %1720, %1723 : tensor<256xf32>
        %1725 = stablehlo.divide %1724, %1704 : tensor<256xf32>
        %1726 = stablehlo.slice %1631 [0:256, 5:6, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1727 = stablehlo.reshape %1726 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1728 = stablehlo.multiply %1667, %1647 : tensor<256xf32>
        %cst_240 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1729 = stablehlo.broadcast_in_dim %cst_240, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1730 = stablehlo.add %1729, %1728 : tensor<256xf32>
        %1731 = stablehlo.subtract %1727, %1730 : tensor<256xf32>
        %1732 = stablehlo.divide %1731, %1704 : tensor<256xf32>
        %1733 = stablehlo.slice %1631 [0:256, 6:7, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1734 = stablehlo.reshape %1733 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1735 = stablehlo.multiply %1672, %1647 : tensor<256xf32>
        %cst_241 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1736 = stablehlo.broadcast_in_dim %cst_241, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1737 = stablehlo.add %1736, %1735 : tensor<256xf32>
        %1738 = stablehlo.subtract %1734, %1737 : tensor<256xf32>
        %1739 = stablehlo.divide %1738, %1704 : tensor<256xf32>
        %1740 = stablehlo.slice %1631 [0:256, 7:8, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1741 = stablehlo.reshape %1740 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1742 = stablehlo.multiply %1677, %1647 : tensor<256xf32>
        %cst_242 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1743 = stablehlo.broadcast_in_dim %cst_242, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1744 = stablehlo.add %1743, %1742 : tensor<256xf32>
        %1745 = stablehlo.subtract %1741, %1744 : tensor<256xf32>
        %1746 = stablehlo.divide %1745, %1704 : tensor<256xf32>
        %1747 = stablehlo.slice %1631 [0:256, 8:9, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1748 = stablehlo.reshape %1747 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1749 = stablehlo.multiply %1682, %1647 : tensor<256xf32>
        %cst_243 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1750 = stablehlo.broadcast_in_dim %cst_243, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1751 = stablehlo.add %1750, %1749 : tensor<256xf32>
        %1752 = stablehlo.subtract %1748, %1751 : tensor<256xf32>
        %1753 = stablehlo.divide %1752, %1704 : tensor<256xf32>
        %1754 = stablehlo.slice %1631 [0:256, 9:10, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1755 = stablehlo.reshape %1754 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1756 = stablehlo.multiply %1687, %1647 : tensor<256xf32>
        %cst_244 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1757 = stablehlo.broadcast_in_dim %cst_244, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1758 = stablehlo.add %1757, %1756 : tensor<256xf32>
        %1759 = stablehlo.subtract %1755, %1758 : tensor<256xf32>
        %1760 = stablehlo.divide %1759, %1704 : tensor<256xf32>
        %1761 = stablehlo.slice %1631 [0:256, 10:11, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1762 = stablehlo.reshape %1761 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1763 = stablehlo.multiply %1692, %1647 : tensor<256xf32>
        %cst_245 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1764 = stablehlo.broadcast_in_dim %cst_245, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1765 = stablehlo.add %1764, %1763 : tensor<256xf32>
        %1766 = stablehlo.subtract %1762, %1765 : tensor<256xf32>
        %1767 = stablehlo.divide %1766, %1704 : tensor<256xf32>
        %1768 = stablehlo.slice %1631 [0:256, 11:12, 1:2] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1769 = stablehlo.reshape %1768 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1770 = stablehlo.multiply %1697, %1647 : tensor<256xf32>
        %cst_246 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1771 = stablehlo.broadcast_in_dim %cst_246, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1772 = stablehlo.add %1771, %1770 : tensor<256xf32>
        %1773 = stablehlo.subtract %1769, %1772 : tensor<256xf32>
        %1774 = stablehlo.divide %1773, %1704 : tensor<256xf32>
        %1775 = stablehlo.slice %1631 [0:256, 2:3, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1776 = stablehlo.reshape %1775 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1777 = stablehlo.multiply %1652, %1652 : tensor<256xf32>
        %cst_247 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1778 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1779 = stablehlo.add %1778, %1777 : tensor<256xf32>
        %1780 = stablehlo.multiply %1711, %1711 : tensor<256xf32>
        %1781 = stablehlo.add %1779, %1780 : tensor<256xf32>
        %1782 = stablehlo.subtract %1776, %1781 : tensor<256xf32>
        %1783 = stablehlo.sqrt %1782 : tensor<256xf32>
        %1784 = stablehlo.slice %1631 [0:256, 3:4, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1785 = stablehlo.reshape %1784 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1786 = stablehlo.multiply %1657, %1652 : tensor<256xf32>
        %cst_248 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1787 = stablehlo.broadcast_in_dim %cst_248, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1788 = stablehlo.add %1787, %1786 : tensor<256xf32>
        %1789 = stablehlo.multiply %1718, %1711 : tensor<256xf32>
        %1790 = stablehlo.add %1788, %1789 : tensor<256xf32>
        %1791 = stablehlo.subtract %1785, %1790 : tensor<256xf32>
        %1792 = stablehlo.divide %1791, %1783 : tensor<256xf32>
        %1793 = stablehlo.slice %1631 [0:256, 4:5, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1794 = stablehlo.reshape %1793 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1795 = stablehlo.multiply %1662, %1652 : tensor<256xf32>
        %cst_249 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1796 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1797 = stablehlo.add %1796, %1795 : tensor<256xf32>
        %1798 = stablehlo.multiply %1725, %1711 : tensor<256xf32>
        %1799 = stablehlo.add %1797, %1798 : tensor<256xf32>
        %1800 = stablehlo.subtract %1794, %1799 : tensor<256xf32>
        %1801 = stablehlo.divide %1800, %1783 : tensor<256xf32>
        %1802 = stablehlo.slice %1631 [0:256, 5:6, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1803 = stablehlo.reshape %1802 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1804 = stablehlo.multiply %1667, %1652 : tensor<256xf32>
        %cst_250 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1805 = stablehlo.broadcast_in_dim %cst_250, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1806 = stablehlo.add %1805, %1804 : tensor<256xf32>
        %1807 = stablehlo.multiply %1732, %1711 : tensor<256xf32>
        %1808 = stablehlo.add %1806, %1807 : tensor<256xf32>
        %1809 = stablehlo.subtract %1803, %1808 : tensor<256xf32>
        %1810 = stablehlo.divide %1809, %1783 : tensor<256xf32>
        %1811 = stablehlo.slice %1631 [0:256, 6:7, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1812 = stablehlo.reshape %1811 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1813 = stablehlo.multiply %1672, %1652 : tensor<256xf32>
        %cst_251 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1814 = stablehlo.broadcast_in_dim %cst_251, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1815 = stablehlo.add %1814, %1813 : tensor<256xf32>
        %1816 = stablehlo.multiply %1739, %1711 : tensor<256xf32>
        %1817 = stablehlo.add %1815, %1816 : tensor<256xf32>
        %1818 = stablehlo.subtract %1812, %1817 : tensor<256xf32>
        %1819 = stablehlo.divide %1818, %1783 : tensor<256xf32>
        %1820 = stablehlo.slice %1631 [0:256, 7:8, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1821 = stablehlo.reshape %1820 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1822 = stablehlo.multiply %1677, %1652 : tensor<256xf32>
        %cst_252 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1823 = stablehlo.broadcast_in_dim %cst_252, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1824 = stablehlo.add %1823, %1822 : tensor<256xf32>
        %1825 = stablehlo.multiply %1746, %1711 : tensor<256xf32>
        %1826 = stablehlo.add %1824, %1825 : tensor<256xf32>
        %1827 = stablehlo.subtract %1821, %1826 : tensor<256xf32>
        %1828 = stablehlo.divide %1827, %1783 : tensor<256xf32>
        %1829 = stablehlo.slice %1631 [0:256, 8:9, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1830 = stablehlo.reshape %1829 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1831 = stablehlo.multiply %1682, %1652 : tensor<256xf32>
        %cst_253 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1832 = stablehlo.broadcast_in_dim %cst_253, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1833 = stablehlo.add %1832, %1831 : tensor<256xf32>
        %1834 = stablehlo.multiply %1753, %1711 : tensor<256xf32>
        %1835 = stablehlo.add %1833, %1834 : tensor<256xf32>
        %1836 = stablehlo.subtract %1830, %1835 : tensor<256xf32>
        %1837 = stablehlo.divide %1836, %1783 : tensor<256xf32>
        %1838 = stablehlo.slice %1631 [0:256, 9:10, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1839 = stablehlo.reshape %1838 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1840 = stablehlo.multiply %1687, %1652 : tensor<256xf32>
        %cst_254 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1841 = stablehlo.broadcast_in_dim %cst_254, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1842 = stablehlo.add %1841, %1840 : tensor<256xf32>
        %1843 = stablehlo.multiply %1760, %1711 : tensor<256xf32>
        %1844 = stablehlo.add %1842, %1843 : tensor<256xf32>
        %1845 = stablehlo.subtract %1839, %1844 : tensor<256xf32>
        %1846 = stablehlo.divide %1845, %1783 : tensor<256xf32>
        %1847 = stablehlo.slice %1631 [0:256, 10:11, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1848 = stablehlo.reshape %1847 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1849 = stablehlo.multiply %1692, %1652 : tensor<256xf32>
        %cst_255 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1850 = stablehlo.broadcast_in_dim %cst_255, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1851 = stablehlo.add %1850, %1849 : tensor<256xf32>
        %1852 = stablehlo.multiply %1767, %1711 : tensor<256xf32>
        %1853 = stablehlo.add %1851, %1852 : tensor<256xf32>
        %1854 = stablehlo.subtract %1848, %1853 : tensor<256xf32>
        %1855 = stablehlo.divide %1854, %1783 : tensor<256xf32>
        %1856 = stablehlo.slice %1631 [0:256, 11:12, 2:3] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1857 = stablehlo.reshape %1856 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1858 = stablehlo.multiply %1697, %1652 : tensor<256xf32>
        %cst_256 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1859 = stablehlo.broadcast_in_dim %cst_256, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1860 = stablehlo.add %1859, %1858 : tensor<256xf32>
        %1861 = stablehlo.multiply %1774, %1711 : tensor<256xf32>
        %1862 = stablehlo.add %1860, %1861 : tensor<256xf32>
        %1863 = stablehlo.subtract %1857, %1862 : tensor<256xf32>
        %1864 = stablehlo.divide %1863, %1783 : tensor<256xf32>
        %1865 = stablehlo.slice %1631 [0:256, 3:4, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1866 = stablehlo.reshape %1865 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1867 = stablehlo.multiply %1657, %1657 : tensor<256xf32>
        %cst_257 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1868 = stablehlo.broadcast_in_dim %cst_257, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1869 = stablehlo.add %1868, %1867 : tensor<256xf32>
        %1870 = stablehlo.multiply %1718, %1718 : tensor<256xf32>
        %1871 = stablehlo.add %1869, %1870 : tensor<256xf32>
        %1872 = stablehlo.multiply %1792, %1792 : tensor<256xf32>
        %1873 = stablehlo.add %1871, %1872 : tensor<256xf32>
        %1874 = stablehlo.subtract %1866, %1873 : tensor<256xf32>
        %1875 = stablehlo.sqrt %1874 : tensor<256xf32>
        %1876 = stablehlo.slice %1631 [0:256, 4:5, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1877 = stablehlo.reshape %1876 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1878 = stablehlo.multiply %1662, %1657 : tensor<256xf32>
        %cst_258 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1879 = stablehlo.broadcast_in_dim %cst_258, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1880 = stablehlo.add %1879, %1878 : tensor<256xf32>
        %1881 = stablehlo.multiply %1725, %1718 : tensor<256xf32>
        %1882 = stablehlo.add %1880, %1881 : tensor<256xf32>
        %1883 = stablehlo.multiply %1801, %1792 : tensor<256xf32>
        %1884 = stablehlo.add %1882, %1883 : tensor<256xf32>
        %1885 = stablehlo.subtract %1877, %1884 : tensor<256xf32>
        %1886 = stablehlo.divide %1885, %1875 : tensor<256xf32>
        %1887 = stablehlo.slice %1631 [0:256, 5:6, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1888 = stablehlo.reshape %1887 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1889 = stablehlo.multiply %1667, %1657 : tensor<256xf32>
        %cst_259 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1890 = stablehlo.broadcast_in_dim %cst_259, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1891 = stablehlo.add %1890, %1889 : tensor<256xf32>
        %1892 = stablehlo.multiply %1732, %1718 : tensor<256xf32>
        %1893 = stablehlo.add %1891, %1892 : tensor<256xf32>
        %1894 = stablehlo.multiply %1810, %1792 : tensor<256xf32>
        %1895 = stablehlo.add %1893, %1894 : tensor<256xf32>
        %1896 = stablehlo.subtract %1888, %1895 : tensor<256xf32>
        %1897 = stablehlo.divide %1896, %1875 : tensor<256xf32>
        %1898 = stablehlo.slice %1631 [0:256, 6:7, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1899 = stablehlo.reshape %1898 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1900 = stablehlo.multiply %1672, %1657 : tensor<256xf32>
        %cst_260 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1901 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1902 = stablehlo.add %1901, %1900 : tensor<256xf32>
        %1903 = stablehlo.multiply %1739, %1718 : tensor<256xf32>
        %1904 = stablehlo.add %1902, %1903 : tensor<256xf32>
        %1905 = stablehlo.multiply %1819, %1792 : tensor<256xf32>
        %1906 = stablehlo.add %1904, %1905 : tensor<256xf32>
        %1907 = stablehlo.subtract %1899, %1906 : tensor<256xf32>
        %1908 = stablehlo.divide %1907, %1875 : tensor<256xf32>
        %1909 = stablehlo.slice %1631 [0:256, 7:8, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1910 = stablehlo.reshape %1909 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1911 = stablehlo.multiply %1677, %1657 : tensor<256xf32>
        %cst_261 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1912 = stablehlo.broadcast_in_dim %cst_261, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1913 = stablehlo.add %1912, %1911 : tensor<256xf32>
        %1914 = stablehlo.multiply %1746, %1718 : tensor<256xf32>
        %1915 = stablehlo.add %1913, %1914 : tensor<256xf32>
        %1916 = stablehlo.multiply %1828, %1792 : tensor<256xf32>
        %1917 = stablehlo.add %1915, %1916 : tensor<256xf32>
        %1918 = stablehlo.subtract %1910, %1917 : tensor<256xf32>
        %1919 = stablehlo.divide %1918, %1875 : tensor<256xf32>
        %1920 = stablehlo.slice %1631 [0:256, 8:9, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1921 = stablehlo.reshape %1920 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1922 = stablehlo.multiply %1682, %1657 : tensor<256xf32>
        %cst_262 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1923 = stablehlo.broadcast_in_dim %cst_262, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1924 = stablehlo.add %1923, %1922 : tensor<256xf32>
        %1925 = stablehlo.multiply %1753, %1718 : tensor<256xf32>
        %1926 = stablehlo.add %1924, %1925 : tensor<256xf32>
        %1927 = stablehlo.multiply %1837, %1792 : tensor<256xf32>
        %1928 = stablehlo.add %1926, %1927 : tensor<256xf32>
        %1929 = stablehlo.subtract %1921, %1928 : tensor<256xf32>
        %1930 = stablehlo.divide %1929, %1875 : tensor<256xf32>
        %1931 = stablehlo.slice %1631 [0:256, 9:10, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1932 = stablehlo.reshape %1931 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1933 = stablehlo.multiply %1687, %1657 : tensor<256xf32>
        %cst_263 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1934 = stablehlo.broadcast_in_dim %cst_263, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1935 = stablehlo.add %1934, %1933 : tensor<256xf32>
        %1936 = stablehlo.multiply %1760, %1718 : tensor<256xf32>
        %1937 = stablehlo.add %1935, %1936 : tensor<256xf32>
        %1938 = stablehlo.multiply %1846, %1792 : tensor<256xf32>
        %1939 = stablehlo.add %1937, %1938 : tensor<256xf32>
        %1940 = stablehlo.subtract %1932, %1939 : tensor<256xf32>
        %1941 = stablehlo.divide %1940, %1875 : tensor<256xf32>
        %1942 = stablehlo.slice %1631 [0:256, 10:11, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1943 = stablehlo.reshape %1942 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1944 = stablehlo.multiply %1692, %1657 : tensor<256xf32>
        %cst_264 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1945 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1946 = stablehlo.add %1945, %1944 : tensor<256xf32>
        %1947 = stablehlo.multiply %1767, %1718 : tensor<256xf32>
        %1948 = stablehlo.add %1946, %1947 : tensor<256xf32>
        %1949 = stablehlo.multiply %1855, %1792 : tensor<256xf32>
        %1950 = stablehlo.add %1948, %1949 : tensor<256xf32>
        %1951 = stablehlo.subtract %1943, %1950 : tensor<256xf32>
        %1952 = stablehlo.divide %1951, %1875 : tensor<256xf32>
        %1953 = stablehlo.slice %1631 [0:256, 11:12, 3:4] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1954 = stablehlo.reshape %1953 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1955 = stablehlo.multiply %1697, %1657 : tensor<256xf32>
        %cst_265 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1956 = stablehlo.broadcast_in_dim %cst_265, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1957 = stablehlo.add %1956, %1955 : tensor<256xf32>
        %1958 = stablehlo.multiply %1774, %1718 : tensor<256xf32>
        %1959 = stablehlo.add %1957, %1958 : tensor<256xf32>
        %1960 = stablehlo.multiply %1864, %1792 : tensor<256xf32>
        %1961 = stablehlo.add %1959, %1960 : tensor<256xf32>
        %1962 = stablehlo.subtract %1954, %1961 : tensor<256xf32>
        %1963 = stablehlo.divide %1962, %1875 : tensor<256xf32>
        %1964 = stablehlo.slice %1631 [0:256, 4:5, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1965 = stablehlo.reshape %1964 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1966 = stablehlo.multiply %1662, %1662 : tensor<256xf32>
        %cst_266 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1967 = stablehlo.broadcast_in_dim %cst_266, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1968 = stablehlo.add %1967, %1966 : tensor<256xf32>
        %1969 = stablehlo.multiply %1725, %1725 : tensor<256xf32>
        %1970 = stablehlo.add %1968, %1969 : tensor<256xf32>
        %1971 = stablehlo.multiply %1801, %1801 : tensor<256xf32>
        %1972 = stablehlo.add %1970, %1971 : tensor<256xf32>
        %1973 = stablehlo.multiply %1886, %1886 : tensor<256xf32>
        %1974 = stablehlo.add %1972, %1973 : tensor<256xf32>
        %1975 = stablehlo.subtract %1965, %1974 : tensor<256xf32>
        %1976 = stablehlo.sqrt %1975 : tensor<256xf32>
        %1977 = stablehlo.slice %1631 [0:256, 5:6, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1978 = stablehlo.reshape %1977 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1979 = stablehlo.multiply %1667, %1662 : tensor<256xf32>
        %cst_267 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1980 = stablehlo.broadcast_in_dim %cst_267, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1981 = stablehlo.add %1980, %1979 : tensor<256xf32>
        %1982 = stablehlo.multiply %1732, %1725 : tensor<256xf32>
        %1983 = stablehlo.add %1981, %1982 : tensor<256xf32>
        %1984 = stablehlo.multiply %1810, %1801 : tensor<256xf32>
        %1985 = stablehlo.add %1983, %1984 : tensor<256xf32>
        %1986 = stablehlo.multiply %1897, %1886 : tensor<256xf32>
        %1987 = stablehlo.add %1985, %1986 : tensor<256xf32>
        %1988 = stablehlo.subtract %1978, %1987 : tensor<256xf32>
        %1989 = stablehlo.divide %1988, %1976 : tensor<256xf32>
        %1990 = stablehlo.slice %1631 [0:256, 6:7, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %1991 = stablehlo.reshape %1990 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %1992 = stablehlo.multiply %1672, %1662 : tensor<256xf32>
        %cst_268 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %1993 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %1994 = stablehlo.add %1993, %1992 : tensor<256xf32>
        %1995 = stablehlo.multiply %1739, %1725 : tensor<256xf32>
        %1996 = stablehlo.add %1994, %1995 : tensor<256xf32>
        %1997 = stablehlo.multiply %1819, %1801 : tensor<256xf32>
        %1998 = stablehlo.add %1996, %1997 : tensor<256xf32>
        %1999 = stablehlo.multiply %1908, %1886 : tensor<256xf32>
        %2000 = stablehlo.add %1998, %1999 : tensor<256xf32>
        %2001 = stablehlo.subtract %1991, %2000 : tensor<256xf32>
        %2002 = stablehlo.divide %2001, %1976 : tensor<256xf32>
        %2003 = stablehlo.slice %1631 [0:256, 7:8, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2004 = stablehlo.reshape %2003 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2005 = stablehlo.multiply %1677, %1662 : tensor<256xf32>
        %cst_269 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2006 = stablehlo.broadcast_in_dim %cst_269, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2007 = stablehlo.add %2006, %2005 : tensor<256xf32>
        %2008 = stablehlo.multiply %1746, %1725 : tensor<256xf32>
        %2009 = stablehlo.add %2007, %2008 : tensor<256xf32>
        %2010 = stablehlo.multiply %1828, %1801 : tensor<256xf32>
        %2011 = stablehlo.add %2009, %2010 : tensor<256xf32>
        %2012 = stablehlo.multiply %1919, %1886 : tensor<256xf32>
        %2013 = stablehlo.add %2011, %2012 : tensor<256xf32>
        %2014 = stablehlo.subtract %2004, %2013 : tensor<256xf32>
        %2015 = stablehlo.divide %2014, %1976 : tensor<256xf32>
        %2016 = stablehlo.slice %1631 [0:256, 8:9, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2017 = stablehlo.reshape %2016 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2018 = stablehlo.multiply %1682, %1662 : tensor<256xf32>
        %cst_270 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2019 = stablehlo.broadcast_in_dim %cst_270, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2020 = stablehlo.add %2019, %2018 : tensor<256xf32>
        %2021 = stablehlo.multiply %1753, %1725 : tensor<256xf32>
        %2022 = stablehlo.add %2020, %2021 : tensor<256xf32>
        %2023 = stablehlo.multiply %1837, %1801 : tensor<256xf32>
        %2024 = stablehlo.add %2022, %2023 : tensor<256xf32>
        %2025 = stablehlo.multiply %1930, %1886 : tensor<256xf32>
        %2026 = stablehlo.add %2024, %2025 : tensor<256xf32>
        %2027 = stablehlo.subtract %2017, %2026 : tensor<256xf32>
        %2028 = stablehlo.divide %2027, %1976 : tensor<256xf32>
        %2029 = stablehlo.slice %1631 [0:256, 9:10, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2030 = stablehlo.reshape %2029 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2031 = stablehlo.multiply %1687, %1662 : tensor<256xf32>
        %cst_271 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2032 = stablehlo.broadcast_in_dim %cst_271, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2033 = stablehlo.add %2032, %2031 : tensor<256xf32>
        %2034 = stablehlo.multiply %1760, %1725 : tensor<256xf32>
        %2035 = stablehlo.add %2033, %2034 : tensor<256xf32>
        %2036 = stablehlo.multiply %1846, %1801 : tensor<256xf32>
        %2037 = stablehlo.add %2035, %2036 : tensor<256xf32>
        %2038 = stablehlo.multiply %1941, %1886 : tensor<256xf32>
        %2039 = stablehlo.add %2037, %2038 : tensor<256xf32>
        %2040 = stablehlo.subtract %2030, %2039 : tensor<256xf32>
        %2041 = stablehlo.divide %2040, %1976 : tensor<256xf32>
        %2042 = stablehlo.slice %1631 [0:256, 10:11, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2043 = stablehlo.reshape %2042 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2044 = stablehlo.multiply %1692, %1662 : tensor<256xf32>
        %cst_272 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2045 = stablehlo.broadcast_in_dim %cst_272, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2046 = stablehlo.add %2045, %2044 : tensor<256xf32>
        %2047 = stablehlo.multiply %1767, %1725 : tensor<256xf32>
        %2048 = stablehlo.add %2046, %2047 : tensor<256xf32>
        %2049 = stablehlo.multiply %1855, %1801 : tensor<256xf32>
        %2050 = stablehlo.add %2048, %2049 : tensor<256xf32>
        %2051 = stablehlo.multiply %1952, %1886 : tensor<256xf32>
        %2052 = stablehlo.add %2050, %2051 : tensor<256xf32>
        %2053 = stablehlo.subtract %2043, %2052 : tensor<256xf32>
        %2054 = stablehlo.divide %2053, %1976 : tensor<256xf32>
        %2055 = stablehlo.slice %1631 [0:256, 11:12, 4:5] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2056 = stablehlo.reshape %2055 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2057 = stablehlo.multiply %1697, %1662 : tensor<256xf32>
        %cst_273 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2058 = stablehlo.broadcast_in_dim %cst_273, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2059 = stablehlo.add %2058, %2057 : tensor<256xf32>
        %2060 = stablehlo.multiply %1774, %1725 : tensor<256xf32>
        %2061 = stablehlo.add %2059, %2060 : tensor<256xf32>
        %2062 = stablehlo.multiply %1864, %1801 : tensor<256xf32>
        %2063 = stablehlo.add %2061, %2062 : tensor<256xf32>
        %2064 = stablehlo.multiply %1963, %1886 : tensor<256xf32>
        %2065 = stablehlo.add %2063, %2064 : tensor<256xf32>
        %2066 = stablehlo.subtract %2056, %2065 : tensor<256xf32>
        %2067 = stablehlo.divide %2066, %1976 : tensor<256xf32>
        %2068 = stablehlo.slice %1631 [0:256, 5:6, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2069 = stablehlo.reshape %2068 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2070 = stablehlo.multiply %1667, %1667 : tensor<256xf32>
        %cst_274 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2071 = stablehlo.broadcast_in_dim %cst_274, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2072 = stablehlo.add %2071, %2070 : tensor<256xf32>
        %2073 = stablehlo.multiply %1732, %1732 : tensor<256xf32>
        %2074 = stablehlo.add %2072, %2073 : tensor<256xf32>
        %2075 = stablehlo.multiply %1810, %1810 : tensor<256xf32>
        %2076 = stablehlo.add %2074, %2075 : tensor<256xf32>
        %2077 = stablehlo.multiply %1897, %1897 : tensor<256xf32>
        %2078 = stablehlo.add %2076, %2077 : tensor<256xf32>
        %2079 = stablehlo.multiply %1989, %1989 : tensor<256xf32>
        %2080 = stablehlo.add %2078, %2079 : tensor<256xf32>
        %2081 = stablehlo.subtract %2069, %2080 : tensor<256xf32>
        %2082 = stablehlo.sqrt %2081 : tensor<256xf32>
        %2083 = stablehlo.slice %1631 [0:256, 6:7, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2084 = stablehlo.reshape %2083 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2085 = stablehlo.multiply %1672, %1667 : tensor<256xf32>
        %cst_275 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2086 = stablehlo.broadcast_in_dim %cst_275, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2087 = stablehlo.add %2086, %2085 : tensor<256xf32>
        %2088 = stablehlo.multiply %1739, %1732 : tensor<256xf32>
        %2089 = stablehlo.add %2087, %2088 : tensor<256xf32>
        %2090 = stablehlo.multiply %1819, %1810 : tensor<256xf32>
        %2091 = stablehlo.add %2089, %2090 : tensor<256xf32>
        %2092 = stablehlo.multiply %1908, %1897 : tensor<256xf32>
        %2093 = stablehlo.add %2091, %2092 : tensor<256xf32>
        %2094 = stablehlo.multiply %2002, %1989 : tensor<256xf32>
        %2095 = stablehlo.add %2093, %2094 : tensor<256xf32>
        %2096 = stablehlo.subtract %2084, %2095 : tensor<256xf32>
        %2097 = stablehlo.divide %2096, %2082 : tensor<256xf32>
        %2098 = stablehlo.slice %1631 [0:256, 7:8, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2099 = stablehlo.reshape %2098 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2100 = stablehlo.multiply %1677, %1667 : tensor<256xf32>
        %cst_276 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2101 = stablehlo.broadcast_in_dim %cst_276, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2102 = stablehlo.add %2101, %2100 : tensor<256xf32>
        %2103 = stablehlo.multiply %1746, %1732 : tensor<256xf32>
        %2104 = stablehlo.add %2102, %2103 : tensor<256xf32>
        %2105 = stablehlo.multiply %1828, %1810 : tensor<256xf32>
        %2106 = stablehlo.add %2104, %2105 : tensor<256xf32>
        %2107 = stablehlo.multiply %1919, %1897 : tensor<256xf32>
        %2108 = stablehlo.add %2106, %2107 : tensor<256xf32>
        %2109 = stablehlo.multiply %2015, %1989 : tensor<256xf32>
        %2110 = stablehlo.add %2108, %2109 : tensor<256xf32>
        %2111 = stablehlo.subtract %2099, %2110 : tensor<256xf32>
        %2112 = stablehlo.divide %2111, %2082 : tensor<256xf32>
        %2113 = stablehlo.slice %1631 [0:256, 8:9, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2114 = stablehlo.reshape %2113 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2115 = stablehlo.multiply %1682, %1667 : tensor<256xf32>
        %cst_277 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2116 = stablehlo.broadcast_in_dim %cst_277, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2117 = stablehlo.add %2116, %2115 : tensor<256xf32>
        %2118 = stablehlo.multiply %1753, %1732 : tensor<256xf32>
        %2119 = stablehlo.add %2117, %2118 : tensor<256xf32>
        %2120 = stablehlo.multiply %1837, %1810 : tensor<256xf32>
        %2121 = stablehlo.add %2119, %2120 : tensor<256xf32>
        %2122 = stablehlo.multiply %1930, %1897 : tensor<256xf32>
        %2123 = stablehlo.add %2121, %2122 : tensor<256xf32>
        %2124 = stablehlo.multiply %2028, %1989 : tensor<256xf32>
        %2125 = stablehlo.add %2123, %2124 : tensor<256xf32>
        %2126 = stablehlo.subtract %2114, %2125 : tensor<256xf32>
        %2127 = stablehlo.divide %2126, %2082 : tensor<256xf32>
        %2128 = stablehlo.slice %1631 [0:256, 9:10, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2129 = stablehlo.reshape %2128 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2130 = stablehlo.multiply %1687, %1667 : tensor<256xf32>
        %cst_278 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2131 = stablehlo.broadcast_in_dim %cst_278, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2132 = stablehlo.add %2131, %2130 : tensor<256xf32>
        %2133 = stablehlo.multiply %1760, %1732 : tensor<256xf32>
        %2134 = stablehlo.add %2132, %2133 : tensor<256xf32>
        %2135 = stablehlo.multiply %1846, %1810 : tensor<256xf32>
        %2136 = stablehlo.add %2134, %2135 : tensor<256xf32>
        %2137 = stablehlo.multiply %1941, %1897 : tensor<256xf32>
        %2138 = stablehlo.add %2136, %2137 : tensor<256xf32>
        %2139 = stablehlo.multiply %2041, %1989 : tensor<256xf32>
        %2140 = stablehlo.add %2138, %2139 : tensor<256xf32>
        %2141 = stablehlo.subtract %2129, %2140 : tensor<256xf32>
        %2142 = stablehlo.divide %2141, %2082 : tensor<256xf32>
        %2143 = stablehlo.slice %1631 [0:256, 10:11, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2144 = stablehlo.reshape %2143 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2145 = stablehlo.multiply %1692, %1667 : tensor<256xf32>
        %cst_279 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2146 = stablehlo.broadcast_in_dim %cst_279, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2147 = stablehlo.add %2146, %2145 : tensor<256xf32>
        %2148 = stablehlo.multiply %1767, %1732 : tensor<256xf32>
        %2149 = stablehlo.add %2147, %2148 : tensor<256xf32>
        %2150 = stablehlo.multiply %1855, %1810 : tensor<256xf32>
        %2151 = stablehlo.add %2149, %2150 : tensor<256xf32>
        %2152 = stablehlo.multiply %1952, %1897 : tensor<256xf32>
        %2153 = stablehlo.add %2151, %2152 : tensor<256xf32>
        %2154 = stablehlo.multiply %2054, %1989 : tensor<256xf32>
        %2155 = stablehlo.add %2153, %2154 : tensor<256xf32>
        %2156 = stablehlo.subtract %2144, %2155 : tensor<256xf32>
        %2157 = stablehlo.divide %2156, %2082 : tensor<256xf32>
        %2158 = stablehlo.slice %1631 [0:256, 11:12, 5:6] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2159 = stablehlo.reshape %2158 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2160 = stablehlo.multiply %1697, %1667 : tensor<256xf32>
        %cst_280 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2161 = stablehlo.broadcast_in_dim %cst_280, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2162 = stablehlo.add %2161, %2160 : tensor<256xf32>
        %2163 = stablehlo.multiply %1774, %1732 : tensor<256xf32>
        %2164 = stablehlo.add %2162, %2163 : tensor<256xf32>
        %2165 = stablehlo.multiply %1864, %1810 : tensor<256xf32>
        %2166 = stablehlo.add %2164, %2165 : tensor<256xf32>
        %2167 = stablehlo.multiply %1963, %1897 : tensor<256xf32>
        %2168 = stablehlo.add %2166, %2167 : tensor<256xf32>
        %2169 = stablehlo.multiply %2067, %1989 : tensor<256xf32>
        %2170 = stablehlo.add %2168, %2169 : tensor<256xf32>
        %2171 = stablehlo.subtract %2159, %2170 : tensor<256xf32>
        %2172 = stablehlo.divide %2171, %2082 : tensor<256xf32>
        %2173 = stablehlo.slice %1631 [0:256, 6:7, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2174 = stablehlo.reshape %2173 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2175 = stablehlo.multiply %1672, %1672 : tensor<256xf32>
        %cst_281 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2176 = stablehlo.broadcast_in_dim %cst_281, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2177 = stablehlo.add %2176, %2175 : tensor<256xf32>
        %2178 = stablehlo.multiply %1739, %1739 : tensor<256xf32>
        %2179 = stablehlo.add %2177, %2178 : tensor<256xf32>
        %2180 = stablehlo.multiply %1819, %1819 : tensor<256xf32>
        %2181 = stablehlo.add %2179, %2180 : tensor<256xf32>
        %2182 = stablehlo.multiply %1908, %1908 : tensor<256xf32>
        %2183 = stablehlo.add %2181, %2182 : tensor<256xf32>
        %2184 = stablehlo.multiply %2002, %2002 : tensor<256xf32>
        %2185 = stablehlo.add %2183, %2184 : tensor<256xf32>
        %2186 = stablehlo.multiply %2097, %2097 : tensor<256xf32>
        %2187 = stablehlo.add %2185, %2186 : tensor<256xf32>
        %2188 = stablehlo.subtract %2174, %2187 : tensor<256xf32>
        %2189 = stablehlo.sqrt %2188 : tensor<256xf32>
        %2190 = stablehlo.slice %1631 [0:256, 7:8, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2191 = stablehlo.reshape %2190 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2192 = stablehlo.multiply %1677, %1672 : tensor<256xf32>
        %cst_282 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2193 = stablehlo.broadcast_in_dim %cst_282, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2194 = stablehlo.add %2193, %2192 : tensor<256xf32>
        %2195 = stablehlo.multiply %1746, %1739 : tensor<256xf32>
        %2196 = stablehlo.add %2194, %2195 : tensor<256xf32>
        %2197 = stablehlo.multiply %1828, %1819 : tensor<256xf32>
        %2198 = stablehlo.add %2196, %2197 : tensor<256xf32>
        %2199 = stablehlo.multiply %1919, %1908 : tensor<256xf32>
        %2200 = stablehlo.add %2198, %2199 : tensor<256xf32>
        %2201 = stablehlo.multiply %2015, %2002 : tensor<256xf32>
        %2202 = stablehlo.add %2200, %2201 : tensor<256xf32>
        %2203 = stablehlo.multiply %2112, %2097 : tensor<256xf32>
        %2204 = stablehlo.add %2202, %2203 : tensor<256xf32>
        %2205 = stablehlo.subtract %2191, %2204 : tensor<256xf32>
        %2206 = stablehlo.divide %2205, %2189 : tensor<256xf32>
        %2207 = stablehlo.slice %1631 [0:256, 8:9, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2208 = stablehlo.reshape %2207 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2209 = stablehlo.multiply %1682, %1672 : tensor<256xf32>
        %cst_283 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2210 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2211 = stablehlo.add %2210, %2209 : tensor<256xf32>
        %2212 = stablehlo.multiply %1753, %1739 : tensor<256xf32>
        %2213 = stablehlo.add %2211, %2212 : tensor<256xf32>
        %2214 = stablehlo.multiply %1837, %1819 : tensor<256xf32>
        %2215 = stablehlo.add %2213, %2214 : tensor<256xf32>
        %2216 = stablehlo.multiply %1930, %1908 : tensor<256xf32>
        %2217 = stablehlo.add %2215, %2216 : tensor<256xf32>
        %2218 = stablehlo.multiply %2028, %2002 : tensor<256xf32>
        %2219 = stablehlo.add %2217, %2218 : tensor<256xf32>
        %2220 = stablehlo.multiply %2127, %2097 : tensor<256xf32>
        %2221 = stablehlo.add %2219, %2220 : tensor<256xf32>
        %2222 = stablehlo.subtract %2208, %2221 : tensor<256xf32>
        %2223 = stablehlo.divide %2222, %2189 : tensor<256xf32>
        %2224 = stablehlo.slice %1631 [0:256, 9:10, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2225 = stablehlo.reshape %2224 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2226 = stablehlo.multiply %1687, %1672 : tensor<256xf32>
        %cst_284 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2227 = stablehlo.broadcast_in_dim %cst_284, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2228 = stablehlo.add %2227, %2226 : tensor<256xf32>
        %2229 = stablehlo.multiply %1760, %1739 : tensor<256xf32>
        %2230 = stablehlo.add %2228, %2229 : tensor<256xf32>
        %2231 = stablehlo.multiply %1846, %1819 : tensor<256xf32>
        %2232 = stablehlo.add %2230, %2231 : tensor<256xf32>
        %2233 = stablehlo.multiply %1941, %1908 : tensor<256xf32>
        %2234 = stablehlo.add %2232, %2233 : tensor<256xf32>
        %2235 = stablehlo.multiply %2041, %2002 : tensor<256xf32>
        %2236 = stablehlo.add %2234, %2235 : tensor<256xf32>
        %2237 = stablehlo.multiply %2142, %2097 : tensor<256xf32>
        %2238 = stablehlo.add %2236, %2237 : tensor<256xf32>
        %2239 = stablehlo.subtract %2225, %2238 : tensor<256xf32>
        %2240 = stablehlo.divide %2239, %2189 : tensor<256xf32>
        %2241 = stablehlo.slice %1631 [0:256, 10:11, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2242 = stablehlo.reshape %2241 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2243 = stablehlo.multiply %1692, %1672 : tensor<256xf32>
        %cst_285 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2244 = stablehlo.broadcast_in_dim %cst_285, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2245 = stablehlo.add %2244, %2243 : tensor<256xf32>
        %2246 = stablehlo.multiply %1767, %1739 : tensor<256xf32>
        %2247 = stablehlo.add %2245, %2246 : tensor<256xf32>
        %2248 = stablehlo.multiply %1855, %1819 : tensor<256xf32>
        %2249 = stablehlo.add %2247, %2248 : tensor<256xf32>
        %2250 = stablehlo.multiply %1952, %1908 : tensor<256xf32>
        %2251 = stablehlo.add %2249, %2250 : tensor<256xf32>
        %2252 = stablehlo.multiply %2054, %2002 : tensor<256xf32>
        %2253 = stablehlo.add %2251, %2252 : tensor<256xf32>
        %2254 = stablehlo.multiply %2157, %2097 : tensor<256xf32>
        %2255 = stablehlo.add %2253, %2254 : tensor<256xf32>
        %2256 = stablehlo.subtract %2242, %2255 : tensor<256xf32>
        %2257 = stablehlo.divide %2256, %2189 : tensor<256xf32>
        %2258 = stablehlo.slice %1631 [0:256, 11:12, 6:7] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2259 = stablehlo.reshape %2258 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2260 = stablehlo.multiply %1697, %1672 : tensor<256xf32>
        %cst_286 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2261 = stablehlo.broadcast_in_dim %cst_286, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2262 = stablehlo.add %2261, %2260 : tensor<256xf32>
        %2263 = stablehlo.multiply %1774, %1739 : tensor<256xf32>
        %2264 = stablehlo.add %2262, %2263 : tensor<256xf32>
        %2265 = stablehlo.multiply %1864, %1819 : tensor<256xf32>
        %2266 = stablehlo.add %2264, %2265 : tensor<256xf32>
        %2267 = stablehlo.multiply %1963, %1908 : tensor<256xf32>
        %2268 = stablehlo.add %2266, %2267 : tensor<256xf32>
        %2269 = stablehlo.multiply %2067, %2002 : tensor<256xf32>
        %2270 = stablehlo.add %2268, %2269 : tensor<256xf32>
        %2271 = stablehlo.multiply %2172, %2097 : tensor<256xf32>
        %2272 = stablehlo.add %2270, %2271 : tensor<256xf32>
        %2273 = stablehlo.subtract %2259, %2272 : tensor<256xf32>
        %2274 = stablehlo.divide %2273, %2189 : tensor<256xf32>
        %2275 = stablehlo.slice %1631 [0:256, 7:8, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2276 = stablehlo.reshape %2275 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2277 = stablehlo.multiply %1677, %1677 : tensor<256xf32>
        %cst_287 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2278 = stablehlo.broadcast_in_dim %cst_287, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2279 = stablehlo.add %2278, %2277 : tensor<256xf32>
        %2280 = stablehlo.multiply %1746, %1746 : tensor<256xf32>
        %2281 = stablehlo.add %2279, %2280 : tensor<256xf32>
        %2282 = stablehlo.multiply %1828, %1828 : tensor<256xf32>
        %2283 = stablehlo.add %2281, %2282 : tensor<256xf32>
        %2284 = stablehlo.multiply %1919, %1919 : tensor<256xf32>
        %2285 = stablehlo.add %2283, %2284 : tensor<256xf32>
        %2286 = stablehlo.multiply %2015, %2015 : tensor<256xf32>
        %2287 = stablehlo.add %2285, %2286 : tensor<256xf32>
        %2288 = stablehlo.multiply %2112, %2112 : tensor<256xf32>
        %2289 = stablehlo.add %2287, %2288 : tensor<256xf32>
        %2290 = stablehlo.multiply %2206, %2206 : tensor<256xf32>
        %2291 = stablehlo.add %2289, %2290 : tensor<256xf32>
        %2292 = stablehlo.subtract %2276, %2291 : tensor<256xf32>
        %2293 = stablehlo.sqrt %2292 : tensor<256xf32>
        %2294 = stablehlo.slice %1631 [0:256, 8:9, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2295 = stablehlo.reshape %2294 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2296 = stablehlo.multiply %1682, %1677 : tensor<256xf32>
        %cst_288 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2297 = stablehlo.broadcast_in_dim %cst_288, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2298 = stablehlo.add %2297, %2296 : tensor<256xf32>
        %2299 = stablehlo.multiply %1753, %1746 : tensor<256xf32>
        %2300 = stablehlo.add %2298, %2299 : tensor<256xf32>
        %2301 = stablehlo.multiply %1837, %1828 : tensor<256xf32>
        %2302 = stablehlo.add %2300, %2301 : tensor<256xf32>
        %2303 = stablehlo.multiply %1930, %1919 : tensor<256xf32>
        %2304 = stablehlo.add %2302, %2303 : tensor<256xf32>
        %2305 = stablehlo.multiply %2028, %2015 : tensor<256xf32>
        %2306 = stablehlo.add %2304, %2305 : tensor<256xf32>
        %2307 = stablehlo.multiply %2127, %2112 : tensor<256xf32>
        %2308 = stablehlo.add %2306, %2307 : tensor<256xf32>
        %2309 = stablehlo.multiply %2223, %2206 : tensor<256xf32>
        %2310 = stablehlo.add %2308, %2309 : tensor<256xf32>
        %2311 = stablehlo.subtract %2295, %2310 : tensor<256xf32>
        %2312 = stablehlo.divide %2311, %2293 : tensor<256xf32>
        %2313 = stablehlo.slice %1631 [0:256, 9:10, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2314 = stablehlo.reshape %2313 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2315 = stablehlo.multiply %1687, %1677 : tensor<256xf32>
        %cst_289 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2316 = stablehlo.broadcast_in_dim %cst_289, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2317 = stablehlo.add %2316, %2315 : tensor<256xf32>
        %2318 = stablehlo.multiply %1760, %1746 : tensor<256xf32>
        %2319 = stablehlo.add %2317, %2318 : tensor<256xf32>
        %2320 = stablehlo.multiply %1846, %1828 : tensor<256xf32>
        %2321 = stablehlo.add %2319, %2320 : tensor<256xf32>
        %2322 = stablehlo.multiply %1941, %1919 : tensor<256xf32>
        %2323 = stablehlo.add %2321, %2322 : tensor<256xf32>
        %2324 = stablehlo.multiply %2041, %2015 : tensor<256xf32>
        %2325 = stablehlo.add %2323, %2324 : tensor<256xf32>
        %2326 = stablehlo.multiply %2142, %2112 : tensor<256xf32>
        %2327 = stablehlo.add %2325, %2326 : tensor<256xf32>
        %2328 = stablehlo.multiply %2240, %2206 : tensor<256xf32>
        %2329 = stablehlo.add %2327, %2328 : tensor<256xf32>
        %2330 = stablehlo.subtract %2314, %2329 : tensor<256xf32>
        %2331 = stablehlo.divide %2330, %2293 : tensor<256xf32>
        %2332 = stablehlo.slice %1631 [0:256, 10:11, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2333 = stablehlo.reshape %2332 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2334 = stablehlo.multiply %1692, %1677 : tensor<256xf32>
        %cst_290 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2335 = stablehlo.broadcast_in_dim %cst_290, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2336 = stablehlo.add %2335, %2334 : tensor<256xf32>
        %2337 = stablehlo.multiply %1767, %1746 : tensor<256xf32>
        %2338 = stablehlo.add %2336, %2337 : tensor<256xf32>
        %2339 = stablehlo.multiply %1855, %1828 : tensor<256xf32>
        %2340 = stablehlo.add %2338, %2339 : tensor<256xf32>
        %2341 = stablehlo.multiply %1952, %1919 : tensor<256xf32>
        %2342 = stablehlo.add %2340, %2341 : tensor<256xf32>
        %2343 = stablehlo.multiply %2054, %2015 : tensor<256xf32>
        %2344 = stablehlo.add %2342, %2343 : tensor<256xf32>
        %2345 = stablehlo.multiply %2157, %2112 : tensor<256xf32>
        %2346 = stablehlo.add %2344, %2345 : tensor<256xf32>
        %2347 = stablehlo.multiply %2257, %2206 : tensor<256xf32>
        %2348 = stablehlo.add %2346, %2347 : tensor<256xf32>
        %2349 = stablehlo.subtract %2333, %2348 : tensor<256xf32>
        %2350 = stablehlo.divide %2349, %2293 : tensor<256xf32>
        %2351 = stablehlo.slice %1631 [0:256, 11:12, 7:8] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2352 = stablehlo.reshape %2351 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2353 = stablehlo.multiply %1697, %1677 : tensor<256xf32>
        %cst_291 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2354 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2355 = stablehlo.add %2354, %2353 : tensor<256xf32>
        %2356 = stablehlo.multiply %1774, %1746 : tensor<256xf32>
        %2357 = stablehlo.add %2355, %2356 : tensor<256xf32>
        %2358 = stablehlo.multiply %1864, %1828 : tensor<256xf32>
        %2359 = stablehlo.add %2357, %2358 : tensor<256xf32>
        %2360 = stablehlo.multiply %1963, %1919 : tensor<256xf32>
        %2361 = stablehlo.add %2359, %2360 : tensor<256xf32>
        %2362 = stablehlo.multiply %2067, %2015 : tensor<256xf32>
        %2363 = stablehlo.add %2361, %2362 : tensor<256xf32>
        %2364 = stablehlo.multiply %2172, %2112 : tensor<256xf32>
        %2365 = stablehlo.add %2363, %2364 : tensor<256xf32>
        %2366 = stablehlo.multiply %2274, %2206 : tensor<256xf32>
        %2367 = stablehlo.add %2365, %2366 : tensor<256xf32>
        %2368 = stablehlo.subtract %2352, %2367 : tensor<256xf32>
        %2369 = stablehlo.divide %2368, %2293 : tensor<256xf32>
        %2370 = stablehlo.slice %1631 [0:256, 8:9, 8:9] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2371 = stablehlo.reshape %2370 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2372 = stablehlo.multiply %1682, %1682 : tensor<256xf32>
        %cst_292 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2373 = stablehlo.broadcast_in_dim %cst_292, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2374 = stablehlo.add %2373, %2372 : tensor<256xf32>
        %2375 = stablehlo.multiply %1753, %1753 : tensor<256xf32>
        %2376 = stablehlo.add %2374, %2375 : tensor<256xf32>
        %2377 = stablehlo.multiply %1837, %1837 : tensor<256xf32>
        %2378 = stablehlo.add %2376, %2377 : tensor<256xf32>
        %2379 = stablehlo.multiply %1930, %1930 : tensor<256xf32>
        %2380 = stablehlo.add %2378, %2379 : tensor<256xf32>
        %2381 = stablehlo.multiply %2028, %2028 : tensor<256xf32>
        %2382 = stablehlo.add %2380, %2381 : tensor<256xf32>
        %2383 = stablehlo.multiply %2127, %2127 : tensor<256xf32>
        %2384 = stablehlo.add %2382, %2383 : tensor<256xf32>
        %2385 = stablehlo.multiply %2223, %2223 : tensor<256xf32>
        %2386 = stablehlo.add %2384, %2385 : tensor<256xf32>
        %2387 = stablehlo.multiply %2312, %2312 : tensor<256xf32>
        %2388 = stablehlo.add %2386, %2387 : tensor<256xf32>
        %2389 = stablehlo.subtract %2371, %2388 : tensor<256xf32>
        %2390 = stablehlo.sqrt %2389 : tensor<256xf32>
        %2391 = stablehlo.slice %1631 [0:256, 9:10, 8:9] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2392 = stablehlo.reshape %2391 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2393 = stablehlo.multiply %1687, %1682 : tensor<256xf32>
        %cst_293 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2394 = stablehlo.broadcast_in_dim %cst_293, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2395 = stablehlo.add %2394, %2393 : tensor<256xf32>
        %2396 = stablehlo.multiply %1760, %1753 : tensor<256xf32>
        %2397 = stablehlo.add %2395, %2396 : tensor<256xf32>
        %2398 = stablehlo.multiply %1846, %1837 : tensor<256xf32>
        %2399 = stablehlo.add %2397, %2398 : tensor<256xf32>
        %2400 = stablehlo.multiply %1941, %1930 : tensor<256xf32>
        %2401 = stablehlo.add %2399, %2400 : tensor<256xf32>
        %2402 = stablehlo.multiply %2041, %2028 : tensor<256xf32>
        %2403 = stablehlo.add %2401, %2402 : tensor<256xf32>
        %2404 = stablehlo.multiply %2142, %2127 : tensor<256xf32>
        %2405 = stablehlo.add %2403, %2404 : tensor<256xf32>
        %2406 = stablehlo.multiply %2240, %2223 : tensor<256xf32>
        %2407 = stablehlo.add %2405, %2406 : tensor<256xf32>
        %2408 = stablehlo.multiply %2331, %2312 : tensor<256xf32>
        %2409 = stablehlo.add %2407, %2408 : tensor<256xf32>
        %2410 = stablehlo.subtract %2392, %2409 : tensor<256xf32>
        %2411 = stablehlo.divide %2410, %2390 : tensor<256xf32>
        %2412 = stablehlo.slice %1631 [0:256, 10:11, 8:9] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2413 = stablehlo.reshape %2412 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2414 = stablehlo.multiply %1692, %1682 : tensor<256xf32>
        %cst_294 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2415 = stablehlo.broadcast_in_dim %cst_294, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2416 = stablehlo.add %2415, %2414 : tensor<256xf32>
        %2417 = stablehlo.multiply %1767, %1753 : tensor<256xf32>
        %2418 = stablehlo.add %2416, %2417 : tensor<256xf32>
        %2419 = stablehlo.multiply %1855, %1837 : tensor<256xf32>
        %2420 = stablehlo.add %2418, %2419 : tensor<256xf32>
        %2421 = stablehlo.multiply %1952, %1930 : tensor<256xf32>
        %2422 = stablehlo.add %2420, %2421 : tensor<256xf32>
        %2423 = stablehlo.multiply %2054, %2028 : tensor<256xf32>
        %2424 = stablehlo.add %2422, %2423 : tensor<256xf32>
        %2425 = stablehlo.multiply %2157, %2127 : tensor<256xf32>
        %2426 = stablehlo.add %2424, %2425 : tensor<256xf32>
        %2427 = stablehlo.multiply %2257, %2223 : tensor<256xf32>
        %2428 = stablehlo.add %2426, %2427 : tensor<256xf32>
        %2429 = stablehlo.multiply %2350, %2312 : tensor<256xf32>
        %2430 = stablehlo.add %2428, %2429 : tensor<256xf32>
        %2431 = stablehlo.subtract %2413, %2430 : tensor<256xf32>
        %2432 = stablehlo.divide %2431, %2390 : tensor<256xf32>
        %2433 = stablehlo.slice %1631 [0:256, 11:12, 8:9] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2434 = stablehlo.reshape %2433 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2435 = stablehlo.multiply %1697, %1682 : tensor<256xf32>
        %cst_295 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2436 = stablehlo.broadcast_in_dim %cst_295, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2437 = stablehlo.add %2436, %2435 : tensor<256xf32>
        %2438 = stablehlo.multiply %1774, %1753 : tensor<256xf32>
        %2439 = stablehlo.add %2437, %2438 : tensor<256xf32>
        %2440 = stablehlo.multiply %1864, %1837 : tensor<256xf32>
        %2441 = stablehlo.add %2439, %2440 : tensor<256xf32>
        %2442 = stablehlo.multiply %1963, %1930 : tensor<256xf32>
        %2443 = stablehlo.add %2441, %2442 : tensor<256xf32>
        %2444 = stablehlo.multiply %2067, %2028 : tensor<256xf32>
        %2445 = stablehlo.add %2443, %2444 : tensor<256xf32>
        %2446 = stablehlo.multiply %2172, %2127 : tensor<256xf32>
        %2447 = stablehlo.add %2445, %2446 : tensor<256xf32>
        %2448 = stablehlo.multiply %2274, %2223 : tensor<256xf32>
        %2449 = stablehlo.add %2447, %2448 : tensor<256xf32>
        %2450 = stablehlo.multiply %2369, %2312 : tensor<256xf32>
        %2451 = stablehlo.add %2449, %2450 : tensor<256xf32>
        %2452 = stablehlo.subtract %2434, %2451 : tensor<256xf32>
        %2453 = stablehlo.divide %2452, %2390 : tensor<256xf32>
        %2454 = stablehlo.slice %1631 [0:256, 9:10, 9:10] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2455 = stablehlo.reshape %2454 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2456 = stablehlo.multiply %1687, %1687 : tensor<256xf32>
        %cst_296 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2457 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2458 = stablehlo.add %2457, %2456 : tensor<256xf32>
        %2459 = stablehlo.multiply %1760, %1760 : tensor<256xf32>
        %2460 = stablehlo.add %2458, %2459 : tensor<256xf32>
        %2461 = stablehlo.multiply %1846, %1846 : tensor<256xf32>
        %2462 = stablehlo.add %2460, %2461 : tensor<256xf32>
        %2463 = stablehlo.multiply %1941, %1941 : tensor<256xf32>
        %2464 = stablehlo.add %2462, %2463 : tensor<256xf32>
        %2465 = stablehlo.multiply %2041, %2041 : tensor<256xf32>
        %2466 = stablehlo.add %2464, %2465 : tensor<256xf32>
        %2467 = stablehlo.multiply %2142, %2142 : tensor<256xf32>
        %2468 = stablehlo.add %2466, %2467 : tensor<256xf32>
        %2469 = stablehlo.multiply %2240, %2240 : tensor<256xf32>
        %2470 = stablehlo.add %2468, %2469 : tensor<256xf32>
        %2471 = stablehlo.multiply %2331, %2331 : tensor<256xf32>
        %2472 = stablehlo.add %2470, %2471 : tensor<256xf32>
        %2473 = stablehlo.multiply %2411, %2411 : tensor<256xf32>
        %2474 = stablehlo.add %2472, %2473 : tensor<256xf32>
        %2475 = stablehlo.subtract %2455, %2474 : tensor<256xf32>
        %2476 = stablehlo.sqrt %2475 : tensor<256xf32>
        %2477 = stablehlo.slice %1631 [0:256, 10:11, 9:10] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2478 = stablehlo.reshape %2477 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2479 = stablehlo.multiply %1692, %1687 : tensor<256xf32>
        %cst_297 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2480 = stablehlo.broadcast_in_dim %cst_297, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2481 = stablehlo.add %2480, %2479 : tensor<256xf32>
        %2482 = stablehlo.multiply %1767, %1760 : tensor<256xf32>
        %2483 = stablehlo.add %2481, %2482 : tensor<256xf32>
        %2484 = stablehlo.multiply %1855, %1846 : tensor<256xf32>
        %2485 = stablehlo.add %2483, %2484 : tensor<256xf32>
        %2486 = stablehlo.multiply %1952, %1941 : tensor<256xf32>
        %2487 = stablehlo.add %2485, %2486 : tensor<256xf32>
        %2488 = stablehlo.multiply %2054, %2041 : tensor<256xf32>
        %2489 = stablehlo.add %2487, %2488 : tensor<256xf32>
        %2490 = stablehlo.multiply %2157, %2142 : tensor<256xf32>
        %2491 = stablehlo.add %2489, %2490 : tensor<256xf32>
        %2492 = stablehlo.multiply %2257, %2240 : tensor<256xf32>
        %2493 = stablehlo.add %2491, %2492 : tensor<256xf32>
        %2494 = stablehlo.multiply %2350, %2331 : tensor<256xf32>
        %2495 = stablehlo.add %2493, %2494 : tensor<256xf32>
        %2496 = stablehlo.multiply %2432, %2411 : tensor<256xf32>
        %2497 = stablehlo.add %2495, %2496 : tensor<256xf32>
        %2498 = stablehlo.subtract %2478, %2497 : tensor<256xf32>
        %2499 = stablehlo.divide %2498, %2476 : tensor<256xf32>
        %2500 = stablehlo.slice %1631 [0:256, 11:12, 9:10] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2501 = stablehlo.reshape %2500 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2502 = stablehlo.multiply %1697, %1687 : tensor<256xf32>
        %cst_298 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2503 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2504 = stablehlo.add %2503, %2502 : tensor<256xf32>
        %2505 = stablehlo.multiply %1774, %1760 : tensor<256xf32>
        %2506 = stablehlo.add %2504, %2505 : tensor<256xf32>
        %2507 = stablehlo.multiply %1864, %1846 : tensor<256xf32>
        %2508 = stablehlo.add %2506, %2507 : tensor<256xf32>
        %2509 = stablehlo.multiply %1963, %1941 : tensor<256xf32>
        %2510 = stablehlo.add %2508, %2509 : tensor<256xf32>
        %2511 = stablehlo.multiply %2067, %2041 : tensor<256xf32>
        %2512 = stablehlo.add %2510, %2511 : tensor<256xf32>
        %2513 = stablehlo.multiply %2172, %2142 : tensor<256xf32>
        %2514 = stablehlo.add %2512, %2513 : tensor<256xf32>
        %2515 = stablehlo.multiply %2274, %2240 : tensor<256xf32>
        %2516 = stablehlo.add %2514, %2515 : tensor<256xf32>
        %2517 = stablehlo.multiply %2369, %2331 : tensor<256xf32>
        %2518 = stablehlo.add %2516, %2517 : tensor<256xf32>
        %2519 = stablehlo.multiply %2453, %2411 : tensor<256xf32>
        %2520 = stablehlo.add %2518, %2519 : tensor<256xf32>
        %2521 = stablehlo.subtract %2501, %2520 : tensor<256xf32>
        %2522 = stablehlo.divide %2521, %2476 : tensor<256xf32>
        %2523 = stablehlo.slice %1631 [0:256, 10:11, 10:11] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2524 = stablehlo.reshape %2523 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2525 = stablehlo.multiply %1692, %1692 : tensor<256xf32>
        %cst_299 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2526 = stablehlo.broadcast_in_dim %cst_299, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2527 = stablehlo.add %2526, %2525 : tensor<256xf32>
        %2528 = stablehlo.multiply %1767, %1767 : tensor<256xf32>
        %2529 = stablehlo.add %2527, %2528 : tensor<256xf32>
        %2530 = stablehlo.multiply %1855, %1855 : tensor<256xf32>
        %2531 = stablehlo.add %2529, %2530 : tensor<256xf32>
        %2532 = stablehlo.multiply %1952, %1952 : tensor<256xf32>
        %2533 = stablehlo.add %2531, %2532 : tensor<256xf32>
        %2534 = stablehlo.multiply %2054, %2054 : tensor<256xf32>
        %2535 = stablehlo.add %2533, %2534 : tensor<256xf32>
        %2536 = stablehlo.multiply %2157, %2157 : tensor<256xf32>
        %2537 = stablehlo.add %2535, %2536 : tensor<256xf32>
        %2538 = stablehlo.multiply %2257, %2257 : tensor<256xf32>
        %2539 = stablehlo.add %2537, %2538 : tensor<256xf32>
        %2540 = stablehlo.multiply %2350, %2350 : tensor<256xf32>
        %2541 = stablehlo.add %2539, %2540 : tensor<256xf32>
        %2542 = stablehlo.multiply %2432, %2432 : tensor<256xf32>
        %2543 = stablehlo.add %2541, %2542 : tensor<256xf32>
        %2544 = stablehlo.multiply %2499, %2499 : tensor<256xf32>
        %2545 = stablehlo.add %2543, %2544 : tensor<256xf32>
        %2546 = stablehlo.subtract %2524, %2545 : tensor<256xf32>
        %2547 = stablehlo.sqrt %2546 : tensor<256xf32>
        %2548 = stablehlo.slice %1631 [0:256, 11:12, 10:11] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2549 = stablehlo.reshape %2548 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2550 = stablehlo.multiply %1697, %1692 : tensor<256xf32>
        %cst_300 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2551 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2552 = stablehlo.add %2551, %2550 : tensor<256xf32>
        %2553 = stablehlo.multiply %1774, %1767 : tensor<256xf32>
        %2554 = stablehlo.add %2552, %2553 : tensor<256xf32>
        %2555 = stablehlo.multiply %1864, %1855 : tensor<256xf32>
        %2556 = stablehlo.add %2554, %2555 : tensor<256xf32>
        %2557 = stablehlo.multiply %1963, %1952 : tensor<256xf32>
        %2558 = stablehlo.add %2556, %2557 : tensor<256xf32>
        %2559 = stablehlo.multiply %2067, %2054 : tensor<256xf32>
        %2560 = stablehlo.add %2558, %2559 : tensor<256xf32>
        %2561 = stablehlo.multiply %2172, %2157 : tensor<256xf32>
        %2562 = stablehlo.add %2560, %2561 : tensor<256xf32>
        %2563 = stablehlo.multiply %2274, %2257 : tensor<256xf32>
        %2564 = stablehlo.add %2562, %2563 : tensor<256xf32>
        %2565 = stablehlo.multiply %2369, %2350 : tensor<256xf32>
        %2566 = stablehlo.add %2564, %2565 : tensor<256xf32>
        %2567 = stablehlo.multiply %2453, %2432 : tensor<256xf32>
        %2568 = stablehlo.add %2566, %2567 : tensor<256xf32>
        %2569 = stablehlo.multiply %2522, %2499 : tensor<256xf32>
        %2570 = stablehlo.add %2568, %2569 : tensor<256xf32>
        %2571 = stablehlo.subtract %2549, %2570 : tensor<256xf32>
        %2572 = stablehlo.divide %2571, %2547 : tensor<256xf32>
        %2573 = stablehlo.slice %1631 [0:256, 11:12, 11:12] : (tensor<256x12x12xf32>) -> tensor<256x1x1xf32>
        %2574 = stablehlo.reshape %2573 : (tensor<256x1x1xf32>) -> tensor<256xf32>
        %2575 = stablehlo.multiply %1697, %1697 : tensor<256xf32>
        %cst_301 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2576 = stablehlo.broadcast_in_dim %cst_301, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2577 = stablehlo.add %2576, %2575 : tensor<256xf32>
        %2578 = stablehlo.multiply %1774, %1774 : tensor<256xf32>
        %2579 = stablehlo.add %2577, %2578 : tensor<256xf32>
        %2580 = stablehlo.multiply %1864, %1864 : tensor<256xf32>
        %2581 = stablehlo.add %2579, %2580 : tensor<256xf32>
        %2582 = stablehlo.multiply %1963, %1963 : tensor<256xf32>
        %2583 = stablehlo.add %2581, %2582 : tensor<256xf32>
        %2584 = stablehlo.multiply %2067, %2067 : tensor<256xf32>
        %2585 = stablehlo.add %2583, %2584 : tensor<256xf32>
        %2586 = stablehlo.multiply %2172, %2172 : tensor<256xf32>
        %2587 = stablehlo.add %2585, %2586 : tensor<256xf32>
        %2588 = stablehlo.multiply %2274, %2274 : tensor<256xf32>
        %2589 = stablehlo.add %2587, %2588 : tensor<256xf32>
        %2590 = stablehlo.multiply %2369, %2369 : tensor<256xf32>
        %2591 = stablehlo.add %2589, %2590 : tensor<256xf32>
        %2592 = stablehlo.multiply %2453, %2453 : tensor<256xf32>
        %2593 = stablehlo.add %2591, %2592 : tensor<256xf32>
        %2594 = stablehlo.multiply %2522, %2522 : tensor<256xf32>
        %2595 = stablehlo.add %2593, %2594 : tensor<256xf32>
        %2596 = stablehlo.multiply %2572, %2572 : tensor<256xf32>
        %2597 = stablehlo.add %2595, %2596 : tensor<256xf32>
        %2598 = stablehlo.subtract %2574, %2597 : tensor<256xf32>
        %2599 = stablehlo.sqrt %2598 : tensor<256xf32>
        %2600 = stablehlo.slice %1637 [0:256, 0:1] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2601 = stablehlo.reshape %2600 : (tensor<256x1xf32>) -> tensor<256xf32>
        %cst_302 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2602 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2603 = stablehlo.subtract %2601, %2602 : tensor<256xf32>
        %2604 = stablehlo.divide %2603, %1642 : tensor<256xf32>
        %2605 = stablehlo.slice %1637 [0:256, 1:2] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2606 = stablehlo.reshape %2605 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2607 = stablehlo.multiply %1647, %2604 : tensor<256xf32>
        %cst_303 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2608 = stablehlo.broadcast_in_dim %cst_303, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2609 = stablehlo.add %2608, %2607 : tensor<256xf32>
        %2610 = stablehlo.subtract %2606, %2609 : tensor<256xf32>
        %2611 = stablehlo.divide %2610, %1704 : tensor<256xf32>
        %2612 = stablehlo.slice %1637 [0:256, 2:3] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2613 = stablehlo.reshape %2612 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2614 = stablehlo.multiply %1652, %2604 : tensor<256xf32>
        %cst_304 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2615 = stablehlo.broadcast_in_dim %cst_304, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2616 = stablehlo.add %2615, %2614 : tensor<256xf32>
        %2617 = stablehlo.multiply %1711, %2611 : tensor<256xf32>
        %2618 = stablehlo.add %2616, %2617 : tensor<256xf32>
        %2619 = stablehlo.subtract %2613, %2618 : tensor<256xf32>
        %2620 = stablehlo.divide %2619, %1783 : tensor<256xf32>
        %2621 = stablehlo.slice %1637 [0:256, 3:4] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2622 = stablehlo.reshape %2621 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2623 = stablehlo.multiply %1657, %2604 : tensor<256xf32>
        %cst_305 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2624 = stablehlo.broadcast_in_dim %cst_305, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2625 = stablehlo.add %2624, %2623 : tensor<256xf32>
        %2626 = stablehlo.multiply %1718, %2611 : tensor<256xf32>
        %2627 = stablehlo.add %2625, %2626 : tensor<256xf32>
        %2628 = stablehlo.multiply %1792, %2620 : tensor<256xf32>
        %2629 = stablehlo.add %2627, %2628 : tensor<256xf32>
        %2630 = stablehlo.subtract %2622, %2629 : tensor<256xf32>
        %2631 = stablehlo.divide %2630, %1875 : tensor<256xf32>
        %2632 = stablehlo.slice %1637 [0:256, 4:5] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2633 = stablehlo.reshape %2632 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2634 = stablehlo.multiply %1662, %2604 : tensor<256xf32>
        %cst_306 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2635 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2636 = stablehlo.add %2635, %2634 : tensor<256xf32>
        %2637 = stablehlo.multiply %1725, %2611 : tensor<256xf32>
        %2638 = stablehlo.add %2636, %2637 : tensor<256xf32>
        %2639 = stablehlo.multiply %1801, %2620 : tensor<256xf32>
        %2640 = stablehlo.add %2638, %2639 : tensor<256xf32>
        %2641 = stablehlo.multiply %1886, %2631 : tensor<256xf32>
        %2642 = stablehlo.add %2640, %2641 : tensor<256xf32>
        %2643 = stablehlo.subtract %2633, %2642 : tensor<256xf32>
        %2644 = stablehlo.divide %2643, %1976 : tensor<256xf32>
        %2645 = stablehlo.slice %1637 [0:256, 5:6] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2646 = stablehlo.reshape %2645 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2647 = stablehlo.multiply %1667, %2604 : tensor<256xf32>
        %cst_307 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2648 = stablehlo.broadcast_in_dim %cst_307, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2649 = stablehlo.add %2648, %2647 : tensor<256xf32>
        %2650 = stablehlo.multiply %1732, %2611 : tensor<256xf32>
        %2651 = stablehlo.add %2649, %2650 : tensor<256xf32>
        %2652 = stablehlo.multiply %1810, %2620 : tensor<256xf32>
        %2653 = stablehlo.add %2651, %2652 : tensor<256xf32>
        %2654 = stablehlo.multiply %1897, %2631 : tensor<256xf32>
        %2655 = stablehlo.add %2653, %2654 : tensor<256xf32>
        %2656 = stablehlo.multiply %1989, %2644 : tensor<256xf32>
        %2657 = stablehlo.add %2655, %2656 : tensor<256xf32>
        %2658 = stablehlo.subtract %2646, %2657 : tensor<256xf32>
        %2659 = stablehlo.divide %2658, %2082 : tensor<256xf32>
        %2660 = stablehlo.slice %1637 [0:256, 6:7] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2661 = stablehlo.reshape %2660 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2662 = stablehlo.multiply %1672, %2604 : tensor<256xf32>
        %cst_308 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2663 = stablehlo.broadcast_in_dim %cst_308, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2664 = stablehlo.add %2663, %2662 : tensor<256xf32>
        %2665 = stablehlo.multiply %1739, %2611 : tensor<256xf32>
        %2666 = stablehlo.add %2664, %2665 : tensor<256xf32>
        %2667 = stablehlo.multiply %1819, %2620 : tensor<256xf32>
        %2668 = stablehlo.add %2666, %2667 : tensor<256xf32>
        %2669 = stablehlo.multiply %1908, %2631 : tensor<256xf32>
        %2670 = stablehlo.add %2668, %2669 : tensor<256xf32>
        %2671 = stablehlo.multiply %2002, %2644 : tensor<256xf32>
        %2672 = stablehlo.add %2670, %2671 : tensor<256xf32>
        %2673 = stablehlo.multiply %2097, %2659 : tensor<256xf32>
        %2674 = stablehlo.add %2672, %2673 : tensor<256xf32>
        %2675 = stablehlo.subtract %2661, %2674 : tensor<256xf32>
        %2676 = stablehlo.divide %2675, %2189 : tensor<256xf32>
        %2677 = stablehlo.slice %1637 [0:256, 7:8] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2678 = stablehlo.reshape %2677 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2679 = stablehlo.multiply %1677, %2604 : tensor<256xf32>
        %cst_309 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2680 = stablehlo.broadcast_in_dim %cst_309, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2681 = stablehlo.add %2680, %2679 : tensor<256xf32>
        %2682 = stablehlo.multiply %1746, %2611 : tensor<256xf32>
        %2683 = stablehlo.add %2681, %2682 : tensor<256xf32>
        %2684 = stablehlo.multiply %1828, %2620 : tensor<256xf32>
        %2685 = stablehlo.add %2683, %2684 : tensor<256xf32>
        %2686 = stablehlo.multiply %1919, %2631 : tensor<256xf32>
        %2687 = stablehlo.add %2685, %2686 : tensor<256xf32>
        %2688 = stablehlo.multiply %2015, %2644 : tensor<256xf32>
        %2689 = stablehlo.add %2687, %2688 : tensor<256xf32>
        %2690 = stablehlo.multiply %2112, %2659 : tensor<256xf32>
        %2691 = stablehlo.add %2689, %2690 : tensor<256xf32>
        %2692 = stablehlo.multiply %2206, %2676 : tensor<256xf32>
        %2693 = stablehlo.add %2691, %2692 : tensor<256xf32>
        %2694 = stablehlo.subtract %2678, %2693 : tensor<256xf32>
        %2695 = stablehlo.divide %2694, %2293 : tensor<256xf32>
        %2696 = stablehlo.slice %1637 [0:256, 8:9] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2697 = stablehlo.reshape %2696 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2698 = stablehlo.multiply %1682, %2604 : tensor<256xf32>
        %cst_310 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2699 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2700 = stablehlo.add %2699, %2698 : tensor<256xf32>
        %2701 = stablehlo.multiply %1753, %2611 : tensor<256xf32>
        %2702 = stablehlo.add %2700, %2701 : tensor<256xf32>
        %2703 = stablehlo.multiply %1837, %2620 : tensor<256xf32>
        %2704 = stablehlo.add %2702, %2703 : tensor<256xf32>
        %2705 = stablehlo.multiply %1930, %2631 : tensor<256xf32>
        %2706 = stablehlo.add %2704, %2705 : tensor<256xf32>
        %2707 = stablehlo.multiply %2028, %2644 : tensor<256xf32>
        %2708 = stablehlo.add %2706, %2707 : tensor<256xf32>
        %2709 = stablehlo.multiply %2127, %2659 : tensor<256xf32>
        %2710 = stablehlo.add %2708, %2709 : tensor<256xf32>
        %2711 = stablehlo.multiply %2223, %2676 : tensor<256xf32>
        %2712 = stablehlo.add %2710, %2711 : tensor<256xf32>
        %2713 = stablehlo.multiply %2312, %2695 : tensor<256xf32>
        %2714 = stablehlo.add %2712, %2713 : tensor<256xf32>
        %2715 = stablehlo.subtract %2697, %2714 : tensor<256xf32>
        %2716 = stablehlo.divide %2715, %2390 : tensor<256xf32>
        %2717 = stablehlo.slice %1637 [0:256, 9:10] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2718 = stablehlo.reshape %2717 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2719 = stablehlo.multiply %1687, %2604 : tensor<256xf32>
        %cst_311 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2720 = stablehlo.broadcast_in_dim %cst_311, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2721 = stablehlo.add %2720, %2719 : tensor<256xf32>
        %2722 = stablehlo.multiply %1760, %2611 : tensor<256xf32>
        %2723 = stablehlo.add %2721, %2722 : tensor<256xf32>
        %2724 = stablehlo.multiply %1846, %2620 : tensor<256xf32>
        %2725 = stablehlo.add %2723, %2724 : tensor<256xf32>
        %2726 = stablehlo.multiply %1941, %2631 : tensor<256xf32>
        %2727 = stablehlo.add %2725, %2726 : tensor<256xf32>
        %2728 = stablehlo.multiply %2041, %2644 : tensor<256xf32>
        %2729 = stablehlo.add %2727, %2728 : tensor<256xf32>
        %2730 = stablehlo.multiply %2142, %2659 : tensor<256xf32>
        %2731 = stablehlo.add %2729, %2730 : tensor<256xf32>
        %2732 = stablehlo.multiply %2240, %2676 : tensor<256xf32>
        %2733 = stablehlo.add %2731, %2732 : tensor<256xf32>
        %2734 = stablehlo.multiply %2331, %2695 : tensor<256xf32>
        %2735 = stablehlo.add %2733, %2734 : tensor<256xf32>
        %2736 = stablehlo.multiply %2411, %2716 : tensor<256xf32>
        %2737 = stablehlo.add %2735, %2736 : tensor<256xf32>
        %2738 = stablehlo.subtract %2718, %2737 : tensor<256xf32>
        %2739 = stablehlo.divide %2738, %2476 : tensor<256xf32>
        %2740 = stablehlo.slice %1637 [0:256, 10:11] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2741 = stablehlo.reshape %2740 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2742 = stablehlo.multiply %1692, %2604 : tensor<256xf32>
        %cst_312 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2743 = stablehlo.broadcast_in_dim %cst_312, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2744 = stablehlo.add %2743, %2742 : tensor<256xf32>
        %2745 = stablehlo.multiply %1767, %2611 : tensor<256xf32>
        %2746 = stablehlo.add %2744, %2745 : tensor<256xf32>
        %2747 = stablehlo.multiply %1855, %2620 : tensor<256xf32>
        %2748 = stablehlo.add %2746, %2747 : tensor<256xf32>
        %2749 = stablehlo.multiply %1952, %2631 : tensor<256xf32>
        %2750 = stablehlo.add %2748, %2749 : tensor<256xf32>
        %2751 = stablehlo.multiply %2054, %2644 : tensor<256xf32>
        %2752 = stablehlo.add %2750, %2751 : tensor<256xf32>
        %2753 = stablehlo.multiply %2157, %2659 : tensor<256xf32>
        %2754 = stablehlo.add %2752, %2753 : tensor<256xf32>
        %2755 = stablehlo.multiply %2257, %2676 : tensor<256xf32>
        %2756 = stablehlo.add %2754, %2755 : tensor<256xf32>
        %2757 = stablehlo.multiply %2350, %2695 : tensor<256xf32>
        %2758 = stablehlo.add %2756, %2757 : tensor<256xf32>
        %2759 = stablehlo.multiply %2432, %2716 : tensor<256xf32>
        %2760 = stablehlo.add %2758, %2759 : tensor<256xf32>
        %2761 = stablehlo.multiply %2499, %2739 : tensor<256xf32>
        %2762 = stablehlo.add %2760, %2761 : tensor<256xf32>
        %2763 = stablehlo.subtract %2741, %2762 : tensor<256xf32>
        %2764 = stablehlo.divide %2763, %2547 : tensor<256xf32>
        %2765 = stablehlo.slice %1637 [0:256, 11:12] : (tensor<256x12xf32>) -> tensor<256x1xf32>
        %2766 = stablehlo.reshape %2765 : (tensor<256x1xf32>) -> tensor<256xf32>
        %2767 = stablehlo.multiply %1697, %2604 : tensor<256xf32>
        %cst_313 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2768 = stablehlo.broadcast_in_dim %cst_313, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2769 = stablehlo.add %2768, %2767 : tensor<256xf32>
        %2770 = stablehlo.multiply %1774, %2611 : tensor<256xf32>
        %2771 = stablehlo.add %2769, %2770 : tensor<256xf32>
        %2772 = stablehlo.multiply %1864, %2620 : tensor<256xf32>
        %2773 = stablehlo.add %2771, %2772 : tensor<256xf32>
        %2774 = stablehlo.multiply %1963, %2631 : tensor<256xf32>
        %2775 = stablehlo.add %2773, %2774 : tensor<256xf32>
        %2776 = stablehlo.multiply %2067, %2644 : tensor<256xf32>
        %2777 = stablehlo.add %2775, %2776 : tensor<256xf32>
        %2778 = stablehlo.multiply %2172, %2659 : tensor<256xf32>
        %2779 = stablehlo.add %2777, %2778 : tensor<256xf32>
        %2780 = stablehlo.multiply %2274, %2676 : tensor<256xf32>
        %2781 = stablehlo.add %2779, %2780 : tensor<256xf32>
        %2782 = stablehlo.multiply %2369, %2695 : tensor<256xf32>
        %2783 = stablehlo.add %2781, %2782 : tensor<256xf32>
        %2784 = stablehlo.multiply %2453, %2716 : tensor<256xf32>
        %2785 = stablehlo.add %2783, %2784 : tensor<256xf32>
        %2786 = stablehlo.multiply %2522, %2739 : tensor<256xf32>
        %2787 = stablehlo.add %2785, %2786 : tensor<256xf32>
        %2788 = stablehlo.multiply %2572, %2764 : tensor<256xf32>
        %2789 = stablehlo.add %2787, %2788 : tensor<256xf32>
        %2790 = stablehlo.subtract %2766, %2789 : tensor<256xf32>
        %2791 = stablehlo.divide %2790, %2599 : tensor<256xf32>
        %cst_314 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2792 = stablehlo.broadcast_in_dim %cst_314, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2793 = stablehlo.subtract %2791, %2792 : tensor<256xf32>
        %2794 = stablehlo.divide %2793, %2599 : tensor<256xf32>
        %2795 = stablehlo.multiply %2572, %2794 : tensor<256xf32>
        %cst_315 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2796 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2797 = stablehlo.add %2796, %2795 : tensor<256xf32>
        %2798 = stablehlo.subtract %2764, %2797 : tensor<256xf32>
        %2799 = stablehlo.divide %2798, %2547 : tensor<256xf32>
        %2800 = stablehlo.multiply %2499, %2799 : tensor<256xf32>
        %cst_316 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2801 = stablehlo.broadcast_in_dim %cst_316, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2802 = stablehlo.add %2801, %2800 : tensor<256xf32>
        %2803 = stablehlo.multiply %2522, %2794 : tensor<256xf32>
        %2804 = stablehlo.add %2802, %2803 : tensor<256xf32>
        %2805 = stablehlo.subtract %2739, %2804 : tensor<256xf32>
        %2806 = stablehlo.divide %2805, %2476 : tensor<256xf32>
        %2807 = stablehlo.multiply %2411, %2806 : tensor<256xf32>
        %cst_317 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2808 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2809 = stablehlo.add %2808, %2807 : tensor<256xf32>
        %2810 = stablehlo.multiply %2432, %2799 : tensor<256xf32>
        %2811 = stablehlo.add %2809, %2810 : tensor<256xf32>
        %2812 = stablehlo.multiply %2453, %2794 : tensor<256xf32>
        %2813 = stablehlo.add %2811, %2812 : tensor<256xf32>
        %2814 = stablehlo.subtract %2716, %2813 : tensor<256xf32>
        %2815 = stablehlo.divide %2814, %2390 : tensor<256xf32>
        %2816 = stablehlo.multiply %2312, %2815 : tensor<256xf32>
        %cst_318 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2817 = stablehlo.broadcast_in_dim %cst_318, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2818 = stablehlo.add %2817, %2816 : tensor<256xf32>
        %2819 = stablehlo.multiply %2331, %2806 : tensor<256xf32>
        %2820 = stablehlo.add %2818, %2819 : tensor<256xf32>
        %2821 = stablehlo.multiply %2350, %2799 : tensor<256xf32>
        %2822 = stablehlo.add %2820, %2821 : tensor<256xf32>
        %2823 = stablehlo.multiply %2369, %2794 : tensor<256xf32>
        %2824 = stablehlo.add %2822, %2823 : tensor<256xf32>
        %2825 = stablehlo.subtract %2695, %2824 : tensor<256xf32>
        %2826 = stablehlo.divide %2825, %2293 : tensor<256xf32>
        %2827 = stablehlo.multiply %2206, %2826 : tensor<256xf32>
        %cst_319 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2828 = stablehlo.broadcast_in_dim %cst_319, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2829 = stablehlo.add %2828, %2827 : tensor<256xf32>
        %2830 = stablehlo.multiply %2223, %2815 : tensor<256xf32>
        %2831 = stablehlo.add %2829, %2830 : tensor<256xf32>
        %2832 = stablehlo.multiply %2240, %2806 : tensor<256xf32>
        %2833 = stablehlo.add %2831, %2832 : tensor<256xf32>
        %2834 = stablehlo.multiply %2257, %2799 : tensor<256xf32>
        %2835 = stablehlo.add %2833, %2834 : tensor<256xf32>
        %2836 = stablehlo.multiply %2274, %2794 : tensor<256xf32>
        %2837 = stablehlo.add %2835, %2836 : tensor<256xf32>
        %2838 = stablehlo.subtract %2676, %2837 : tensor<256xf32>
        %2839 = stablehlo.divide %2838, %2189 : tensor<256xf32>
        %2840 = stablehlo.multiply %2097, %2839 : tensor<256xf32>
        %cst_320 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2841 = stablehlo.broadcast_in_dim %cst_320, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2842 = stablehlo.add %2841, %2840 : tensor<256xf32>
        %2843 = stablehlo.multiply %2112, %2826 : tensor<256xf32>
        %2844 = stablehlo.add %2842, %2843 : tensor<256xf32>
        %2845 = stablehlo.multiply %2127, %2815 : tensor<256xf32>
        %2846 = stablehlo.add %2844, %2845 : tensor<256xf32>
        %2847 = stablehlo.multiply %2142, %2806 : tensor<256xf32>
        %2848 = stablehlo.add %2846, %2847 : tensor<256xf32>
        %2849 = stablehlo.multiply %2157, %2799 : tensor<256xf32>
        %2850 = stablehlo.add %2848, %2849 : tensor<256xf32>
        %2851 = stablehlo.multiply %2172, %2794 : tensor<256xf32>
        %2852 = stablehlo.add %2850, %2851 : tensor<256xf32>
        %2853 = stablehlo.subtract %2659, %2852 : tensor<256xf32>
        %2854 = stablehlo.divide %2853, %2082 : tensor<256xf32>
        %2855 = stablehlo.multiply %1989, %2854 : tensor<256xf32>
        %cst_321 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2856 = stablehlo.broadcast_in_dim %cst_321, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2857 = stablehlo.add %2856, %2855 : tensor<256xf32>
        %2858 = stablehlo.multiply %2002, %2839 : tensor<256xf32>
        %2859 = stablehlo.add %2857, %2858 : tensor<256xf32>
        %2860 = stablehlo.multiply %2015, %2826 : tensor<256xf32>
        %2861 = stablehlo.add %2859, %2860 : tensor<256xf32>
        %2862 = stablehlo.multiply %2028, %2815 : tensor<256xf32>
        %2863 = stablehlo.add %2861, %2862 : tensor<256xf32>
        %2864 = stablehlo.multiply %2041, %2806 : tensor<256xf32>
        %2865 = stablehlo.add %2863, %2864 : tensor<256xf32>
        %2866 = stablehlo.multiply %2054, %2799 : tensor<256xf32>
        %2867 = stablehlo.add %2865, %2866 : tensor<256xf32>
        %2868 = stablehlo.multiply %2067, %2794 : tensor<256xf32>
        %2869 = stablehlo.add %2867, %2868 : tensor<256xf32>
        %2870 = stablehlo.subtract %2644, %2869 : tensor<256xf32>
        %2871 = stablehlo.divide %2870, %1976 : tensor<256xf32>
        %2872 = stablehlo.multiply %1886, %2871 : tensor<256xf32>
        %cst_322 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2873 = stablehlo.broadcast_in_dim %cst_322, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2874 = stablehlo.add %2873, %2872 : tensor<256xf32>
        %2875 = stablehlo.multiply %1897, %2854 : tensor<256xf32>
        %2876 = stablehlo.add %2874, %2875 : tensor<256xf32>
        %2877 = stablehlo.multiply %1908, %2839 : tensor<256xf32>
        %2878 = stablehlo.add %2876, %2877 : tensor<256xf32>
        %2879 = stablehlo.multiply %1919, %2826 : tensor<256xf32>
        %2880 = stablehlo.add %2878, %2879 : tensor<256xf32>
        %2881 = stablehlo.multiply %1930, %2815 : tensor<256xf32>
        %2882 = stablehlo.add %2880, %2881 : tensor<256xf32>
        %2883 = stablehlo.multiply %1941, %2806 : tensor<256xf32>
        %2884 = stablehlo.add %2882, %2883 : tensor<256xf32>
        %2885 = stablehlo.multiply %1952, %2799 : tensor<256xf32>
        %2886 = stablehlo.add %2884, %2885 : tensor<256xf32>
        %2887 = stablehlo.multiply %1963, %2794 : tensor<256xf32>
        %2888 = stablehlo.add %2886, %2887 : tensor<256xf32>
        %2889 = stablehlo.subtract %2631, %2888 : tensor<256xf32>
        %2890 = stablehlo.divide %2889, %1875 : tensor<256xf32>
        %2891 = stablehlo.multiply %1792, %2890 : tensor<256xf32>
        %cst_323 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2892 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2893 = stablehlo.add %2892, %2891 : tensor<256xf32>
        %2894 = stablehlo.multiply %1801, %2871 : tensor<256xf32>
        %2895 = stablehlo.add %2893, %2894 : tensor<256xf32>
        %2896 = stablehlo.multiply %1810, %2854 : tensor<256xf32>
        %2897 = stablehlo.add %2895, %2896 : tensor<256xf32>
        %2898 = stablehlo.multiply %1819, %2839 : tensor<256xf32>
        %2899 = stablehlo.add %2897, %2898 : tensor<256xf32>
        %2900 = stablehlo.multiply %1828, %2826 : tensor<256xf32>
        %2901 = stablehlo.add %2899, %2900 : tensor<256xf32>
        %2902 = stablehlo.multiply %1837, %2815 : tensor<256xf32>
        %2903 = stablehlo.add %2901, %2902 : tensor<256xf32>
        %2904 = stablehlo.multiply %1846, %2806 : tensor<256xf32>
        %2905 = stablehlo.add %2903, %2904 : tensor<256xf32>
        %2906 = stablehlo.multiply %1855, %2799 : tensor<256xf32>
        %2907 = stablehlo.add %2905, %2906 : tensor<256xf32>
        %2908 = stablehlo.multiply %1864, %2794 : tensor<256xf32>
        %2909 = stablehlo.add %2907, %2908 : tensor<256xf32>
        %2910 = stablehlo.subtract %2620, %2909 : tensor<256xf32>
        %2911 = stablehlo.divide %2910, %1783 : tensor<256xf32>
        %2912 = stablehlo.multiply %1711, %2911 : tensor<256xf32>
        %cst_324 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2913 = stablehlo.broadcast_in_dim %cst_324, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2914 = stablehlo.add %2913, %2912 : tensor<256xf32>
        %2915 = stablehlo.multiply %1718, %2890 : tensor<256xf32>
        %2916 = stablehlo.add %2914, %2915 : tensor<256xf32>
        %2917 = stablehlo.multiply %1725, %2871 : tensor<256xf32>
        %2918 = stablehlo.add %2916, %2917 : tensor<256xf32>
        %2919 = stablehlo.multiply %1732, %2854 : tensor<256xf32>
        %2920 = stablehlo.add %2918, %2919 : tensor<256xf32>
        %2921 = stablehlo.multiply %1739, %2839 : tensor<256xf32>
        %2922 = stablehlo.add %2920, %2921 : tensor<256xf32>
        %2923 = stablehlo.multiply %1746, %2826 : tensor<256xf32>
        %2924 = stablehlo.add %2922, %2923 : tensor<256xf32>
        %2925 = stablehlo.multiply %1753, %2815 : tensor<256xf32>
        %2926 = stablehlo.add %2924, %2925 : tensor<256xf32>
        %2927 = stablehlo.multiply %1760, %2806 : tensor<256xf32>
        %2928 = stablehlo.add %2926, %2927 : tensor<256xf32>
        %2929 = stablehlo.multiply %1767, %2799 : tensor<256xf32>
        %2930 = stablehlo.add %2928, %2929 : tensor<256xf32>
        %2931 = stablehlo.multiply %1774, %2794 : tensor<256xf32>
        %2932 = stablehlo.add %2930, %2931 : tensor<256xf32>
        %2933 = stablehlo.subtract %2611, %2932 : tensor<256xf32>
        %2934 = stablehlo.divide %2933, %1704 : tensor<256xf32>
        %2935 = stablehlo.multiply %1647, %2934 : tensor<256xf32>
        %cst_325 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2936 = stablehlo.broadcast_in_dim %cst_325, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %2937 = stablehlo.add %2936, %2935 : tensor<256xf32>
        %2938 = stablehlo.multiply %1652, %2911 : tensor<256xf32>
        %2939 = stablehlo.add %2937, %2938 : tensor<256xf32>
        %2940 = stablehlo.multiply %1657, %2890 : tensor<256xf32>
        %2941 = stablehlo.add %2939, %2940 : tensor<256xf32>
        %2942 = stablehlo.multiply %1662, %2871 : tensor<256xf32>
        %2943 = stablehlo.add %2941, %2942 : tensor<256xf32>
        %2944 = stablehlo.multiply %1667, %2854 : tensor<256xf32>
        %2945 = stablehlo.add %2943, %2944 : tensor<256xf32>
        %2946 = stablehlo.multiply %1672, %2839 : tensor<256xf32>
        %2947 = stablehlo.add %2945, %2946 : tensor<256xf32>
        %2948 = stablehlo.multiply %1677, %2826 : tensor<256xf32>
        %2949 = stablehlo.add %2947, %2948 : tensor<256xf32>
        %2950 = stablehlo.multiply %1682, %2815 : tensor<256xf32>
        %2951 = stablehlo.add %2949, %2950 : tensor<256xf32>
        %2952 = stablehlo.multiply %1687, %2806 : tensor<256xf32>
        %2953 = stablehlo.add %2951, %2952 : tensor<256xf32>
        %2954 = stablehlo.multiply %1692, %2799 : tensor<256xf32>
        %2955 = stablehlo.add %2953, %2954 : tensor<256xf32>
        %2956 = stablehlo.multiply %1697, %2794 : tensor<256xf32>
        %2957 = stablehlo.add %2955, %2956 : tensor<256xf32>
        %2958 = stablehlo.subtract %2604, %2957 : tensor<256xf32>
        %2959 = stablehlo.divide %2958, %1642 : tensor<256xf32>
        %2960 = stablehlo.broadcast_in_dim %2959, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2961 = stablehlo.broadcast_in_dim %2934, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2962 = stablehlo.broadcast_in_dim %2911, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2963 = stablehlo.broadcast_in_dim %2890, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2964 = stablehlo.broadcast_in_dim %2871, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2965 = stablehlo.broadcast_in_dim %2854, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2966 = stablehlo.broadcast_in_dim %2839, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2967 = stablehlo.broadcast_in_dim %2826, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2968 = stablehlo.broadcast_in_dim %2815, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2969 = stablehlo.broadcast_in_dim %2806, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2970 = stablehlo.broadcast_in_dim %2799, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2971 = stablehlo.broadcast_in_dim %2794, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
        %2972 = stablehlo.concatenate %2960, %2961, %2962, %2963, %2964, %2965, %2966, %2967, %2968, %2969, %2970, %2971, dim = 0 : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<12x256xf32>
        %2973 = stablehlo.transpose %2972, dims = [1, 0] : (tensor<12x256xf32>) -> tensor<256x12xf32>
        %2974 = stablehlo.add %iterArg_200, %2973 : tensor<256x12xf32>
        %2975 = stablehlo.dot_general %iterArg_187, %2974, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<8x12xf32>, tensor<256x12xf32>) -> tensor<8x256xf32>
        %2976 = stablehlo.transpose %2975, dims = [1, 0] : (tensor<8x256xf32>) -> tensor<256x8xf32>
        %2977 = stablehlo.add %2976, %iterArg_188 : tensor<256x8xf32>
        %2978 = stablehlo.exponential %2977 : tensor<256x8xf32>
        %cst_326 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2979 = stablehlo.broadcast_in_dim %cst_326, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %2980 = stablehlo.maximum %2978, %2979 : tensor<256x8xf32>
        %cst_327 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2981 = stablehlo.broadcast_in_dim %cst_327, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %2982 = stablehlo.compare NE, %iterArg_190, %2981, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %2983 = stablehlo.compare NE, %iterArg_190, %iterArg_190, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %2984 = stablehlo.or %2982, %2983 : tensor<256x8xi1>
        %2985 = stablehlo.log %iterArg_190 : tensor<256x8xf32>
        %2986 = stablehlo.multiply %iterArg_190, %2985 : tensor<256x8xf32>
        %cst_328 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2987 = stablehlo.broadcast_in_dim %cst_328, dims = [] : (tensor<f32>) -> tensor<8xf32>
        %2988 = func.call @_where_3(%2984, %2986, %2987) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
        %cst_329 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2989 = stablehlo.broadcast_in_dim %cst_329, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %2990 = stablehlo.compare NE, %iterArg_190, %2989, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %2991 = stablehlo.compare NE, %2980, %2980, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %2992 = stablehlo.or %2990, %2991 : tensor<256x8xi1>
        %2993 = stablehlo.log %2980 : tensor<256x8xf32>
        %2994 = stablehlo.multiply %iterArg_190, %2993 : tensor<256x8xf32>
        %cst_330 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %2995 = stablehlo.broadcast_in_dim %cst_330, dims = [] : (tensor<f32>) -> tensor<8xf32>
        %2996 = func.call @_where_3(%2992, %2994, %2995) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
        %2997 = stablehlo.subtract %2988, %2996 : tensor<256x8xf32>
        %2998 = stablehlo.subtract %iterArg_190, %2980 : tensor<256x8xf32>
        %2999 = stablehlo.subtract %2997, %2998 : tensor<256x8xf32>
        %cst_331 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
        %3000 = stablehlo.broadcast_in_dim %cst_331, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %3001 = stablehlo.multiply %3000, %2999 : tensor<256x8xf32>
        %cst_332 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %3002 = stablehlo.broadcast_in_dim %cst_332, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %3003 = stablehlo.compare GT, %iterArg_189, %3002, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
        %cst_333 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
        %3004 = func.call @_where_4(%3003, %iterArg_189, %cst_333) : (tensor<256xi1>, tensor<256xf32>, tensor<f64>) -> tensor<256xf32>
        %cst_334 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %3005 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %3006 = stablehlo.compare NE, %iterArg_190, %3005, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %3007 = stablehlo.compare NE, %iterArg_190, %iterArg_190, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %3008 = stablehlo.or %3006, %3007 : tensor<256x8xi1>
        %3009 = stablehlo.log %iterArg_190 : tensor<256x8xf32>
        %3010 = stablehlo.multiply %iterArg_190, %3009 : tensor<256x8xf32>
        %cst_335 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %3011 = stablehlo.broadcast_in_dim %cst_335, dims = [] : (tensor<f32>) -> tensor<8xf32>
        %3012 = func.call @_where_3(%3008, %3010, %3011) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
        %cst_336 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %3013 = stablehlo.broadcast_in_dim %cst_336, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %3014 = stablehlo.compare NE, %iterArg_190, %3013, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %3015 = stablehlo.compare NE, %2980, %2980, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
        %3016 = stablehlo.or %3014, %3015 : tensor<256x8xi1>
        %3017 = stablehlo.log %2980 : tensor<256x8xf32>
        %3018 = stablehlo.multiply %iterArg_190, %3017 : tensor<256x8xf32>
        %cst_337 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %3019 = stablehlo.broadcast_in_dim %cst_337, dims = [] : (tensor<f32>) -> tensor<8xf32>
        %3020 = func.call @_where_3(%3016, %3018, %3019) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
        %3021 = stablehlo.subtract %3012, %3020 : tensor<256x8xf32>
        %cst_338 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %3022 = stablehlo.broadcast_in_dim %cst_338, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %3023 = stablehlo.divide %3022, %3004 : tensor<256xf32>
        %3024 = stablehlo.broadcast_in_dim %3023, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
        %3025 = stablehlo.broadcast_in_dim %3024, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
        %3026 = stablehlo.add %iterArg_190, %3025 : tensor<256x8xf32>
        %3027 = stablehlo.broadcast_in_dim %3004, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
        %3028 = stablehlo.broadcast_in_dim %3027, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
        %3029 = stablehlo.multiply %3028, %iterArg_190 : tensor<256x8xf32>
        %3030 = stablehlo.log_plus_one %3029 : tensor<256x8xf32>
        %3031 = stablehlo.broadcast_in_dim %3004, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
        %3032 = stablehlo.broadcast_in_dim %3031, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
        %3033 = stablehlo.multiply %3032, %2980 : tensor<256x8xf32>
        %3034 = stablehlo.log_plus_one %3033 : tensor<256x8xf32>
        %3035 = stablehlo.subtract %3030, %3034 : tensor<256x8xf32>
        %3036 = stablehlo.multiply %3026, %3035 : tensor<256x8xf32>
        %3037 = stablehlo.subtract %3021, %3036 : tensor<256x8xf32>
        %cst_339 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
        %3038 = stablehlo.broadcast_in_dim %cst_339, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
        %3039 = stablehlo.multiply %3038, %3037 : tensor<256x8xf32>
        %cst_340 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %3040 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %3041 = stablehlo.compare GT, %iterArg_189, %3040, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
        %3042 = func.call @_where_5(%3041, %3039, %3001) : (tensor<256xi1>, tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xf32>
        %3043 = stablehlo.multiply %iterArg_191, %3042 : tensor<256x8xf32>
        %cst_341 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        %3044 = stablehlo.reduce(%3043 init: %cst_341) applies stablehlo.add across dimensions = [1] : (tensor<256x8xf32>, tensor<f32>) -> tensor<256xf32>
        %c_342 = stablehlo.constant dense<0> : tensor<i64>
        %3045 = stablehlo.broadcast_in_dim %c_342, dims = [] : (tensor<i64>) -> tensor<256xi64>
        %3046:10 = stablehlo.while(%iterArg_353 = %iterArg_201, %iterArg_354 = %iterArg_200, %iterArg_355 = %iterArg_187, %iterArg_356 = %iterArg_188, %iterArg_357 = %iterArg_190, %iterArg_358 = %iterArg_189, %iterArg_359 = %iterArg_191, %iterArg_360 = %2974, %iterArg_361 = %3044, %iterArg_362 = %3045) : tensor<256xf32>, tensor<256x12xf32>, tensor<8x12xf32>, tensor<256x8xf32>, tensor<256x8xf32>, tensor<256xf32>, tensor<256x8xf32>, tensor<256x12xf32>, tensor<256xf32>, tensor<256xi64>
        cond {
          %3086 = stablehlo.is_finite %iterArg_361 : (tensor<256xf32>) -> tensor<256xi1>
          %3087 = stablehlo.not %3086 : tensor<256xi1>
          %cst_363 = stablehlo.constant dense<1.00011921> : tensor<f32>
          %3088 = stablehlo.broadcast_in_dim %cst_363, dims = [] : (tensor<f32>) -> tensor<256xf32>
          %3089 = stablehlo.multiply %iterArg_353, %3088 : tensor<256xf32>
          %cst_364 = stablehlo.constant dense<1.1920929E-4> : tensor<f32>
          %3090 = stablehlo.broadcast_in_dim %cst_364, dims = [] : (tensor<f32>) -> tensor<256xf32>
          %3091 = stablehlo.add %3089, %3090 : tensor<256xf32>
          %3092 = stablehlo.compare GT, %iterArg_361, %3091, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
          %3093 = stablehlo.or %3087, %3092 : tensor<256xi1>
          %c_365 = stablehlo.constant dense<30> : tensor<i64>
          %3094 = stablehlo.broadcast_in_dim %c_365, dims = [] : (tensor<i64>) -> tensor<256xi64>
          %3095 = stablehlo.compare LT, %iterArg_362, %3094, SIGNED : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
          %c_366 = stablehlo.constant dense<false> : tensor<i1>
          %3096 = stablehlo.broadcast_in_dim %c_366, dims = [] : (tensor<i1>) -> tensor<256xi1>
          %3097 = stablehlo.compare NE, %3095, %3096, UNSIGNED : (tensor<256xi1>, tensor<256xi1>) -> tensor<256xi1>
          %3098 = stablehlo.convert %3097 : tensor<256xi1>
          %3099 = stablehlo.and %3093, %3098 : tensor<256xi1>
          %c_367 = stablehlo.constant dense<false> : tensor<i1>
          %3100 = stablehlo.reduce(%3099 init: %c_367) applies stablehlo.or across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
          stablehlo.return %3100 : tensor<i1>
        } do {
          %3086 = stablehlo.add %iterArg_360, %iterArg_354 : tensor<256x12xf32>
          %cst_363 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
          %3087 = stablehlo.broadcast_in_dim %cst_363, dims = [] : (tensor<f32>) -> tensor<256x12xf32>
          %3088 = stablehlo.multiply %3087, %3086 : tensor<256x12xf32>
          %3089 = stablehlo.dot_general %iterArg_355, %3088, contracting_dims = [1] x [1], precision = [HIGHEST, HIGHEST] : (tensor<8x12xf32>, tensor<256x12xf32>) -> tensor<8x256xf32>
          %3090 = stablehlo.transpose %3089, dims = [1, 0] : (tensor<8x256xf32>) -> tensor<256x8xf32>
          %3091 = stablehlo.add %3090, %iterArg_356 : tensor<256x8xf32>
          %3092 = stablehlo.exponential %3091 : tensor<256x8xf32>
          %cst_364 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3093 = stablehlo.broadcast_in_dim %cst_364, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
          %3094 = stablehlo.maximum %3092, %3093 : tensor<256x8xf32>
          %cst_365 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3095 = stablehlo.broadcast_in_dim %cst_365, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
          %3096 = stablehlo.compare NE, %iterArg_357, %3095, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
          %3097 = stablehlo.compare NE, %iterArg_357, %iterArg_357, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
          %3098 = stablehlo.or %3096, %3097 : tensor<256x8xi1>
          %3099 = stablehlo.log %iterArg_357 : tensor<256x8xf32>
          %3100 = stablehlo.multiply %iterArg_357, %3099 : tensor<256x8xf32>
          %cst_366 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3101 = stablehlo.broadcast_in_dim %cst_366, dims = [] : (tensor<f32>) -> tensor<8xf32>
          %3102 = func.call @_where_3(%3098, %3100, %3101) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
          %cst_367 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3103 = stablehlo.broadcast_in_dim %cst_367, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
          %3104 = stablehlo.compare NE, %iterArg_357, %3103, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
          %3105 = stablehlo.compare NE, %3094, %3094, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
          %3106 = stablehlo.or %3104, %3105 : tensor<256x8xi1>
          %3107 = stablehlo.log %3094 : tensor<256x8xf32>
          %3108 = stablehlo.multiply %iterArg_357, %3107 : tensor<256x8xf32>
          %cst_368 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3109 = stablehlo.broadcast_in_dim %cst_368, dims = [] : (tensor<f32>) -> tensor<8xf32>
          %3110 = func.call @_where_3(%3106, %3108, %3109) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
          %3111 = stablehlo.subtract %3102, %3110 : tensor<256x8xf32>
          %3112 = stablehlo.subtract %iterArg_357, %3094 : tensor<256x8xf32>
          %3113 = stablehlo.subtract %3111, %3112 : tensor<256x8xf32>
          %cst_369 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
          %3114 = stablehlo.broadcast_in_dim %cst_369, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
          %3115 = stablehlo.multiply %3114, %3113 : tensor<256x8xf32>
          %cst_370 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3116 = stablehlo.broadcast_in_dim %cst_370, dims = [] : (tensor<f32>) -> tensor<256xf32>
          %3117 = stablehlo.compare GT, %iterArg_358, %3116, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
          %cst_371 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
          %3118 = func.call @_where_4(%3117, %iterArg_358, %cst_371) : (tensor<256xi1>, tensor<256xf32>, tensor<f64>) -> tensor<256xf32>
          %cst_372 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3119 = stablehlo.broadcast_in_dim %cst_372, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
          %3120 = stablehlo.compare NE, %iterArg_357, %3119, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
          %3121 = stablehlo.compare NE, %iterArg_357, %iterArg_357, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
          %3122 = stablehlo.or %3120, %3121 : tensor<256x8xi1>
          %3123 = stablehlo.log %iterArg_357 : tensor<256x8xf32>
          %3124 = stablehlo.multiply %iterArg_357, %3123 : tensor<256x8xf32>
          %cst_373 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3125 = stablehlo.broadcast_in_dim %cst_373, dims = [] : (tensor<f32>) -> tensor<8xf32>
          %3126 = func.call @_where_3(%3122, %3124, %3125) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
          %cst_374 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3127 = stablehlo.broadcast_in_dim %cst_374, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
          %3128 = stablehlo.compare NE, %iterArg_357, %3127, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
          %3129 = stablehlo.compare NE, %3094, %3094, FLOAT : (tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xi1>
          %3130 = stablehlo.or %3128, %3129 : tensor<256x8xi1>
          %3131 = stablehlo.log %3094 : tensor<256x8xf32>
          %3132 = stablehlo.multiply %iterArg_357, %3131 : tensor<256x8xf32>
          %cst_375 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3133 = stablehlo.broadcast_in_dim %cst_375, dims = [] : (tensor<f32>) -> tensor<8xf32>
          %3134 = func.call @_where_3(%3130, %3132, %3133) : (tensor<256x8xi1>, tensor<256x8xf32>, tensor<8xf32>) -> tensor<256x8xf32>
          %3135 = stablehlo.subtract %3126, %3134 : tensor<256x8xf32>
          %cst_376 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
          %3136 = stablehlo.broadcast_in_dim %cst_376, dims = [] : (tensor<f32>) -> tensor<256xf32>
          %3137 = stablehlo.divide %3136, %3118 : tensor<256xf32>
          %3138 = stablehlo.broadcast_in_dim %3137, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
          %3139 = stablehlo.broadcast_in_dim %3138, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
          %3140 = stablehlo.add %iterArg_357, %3139 : tensor<256x8xf32>
          %3141 = stablehlo.broadcast_in_dim %3118, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
          %3142 = stablehlo.broadcast_in_dim %3141, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
          %3143 = stablehlo.multiply %3142, %iterArg_357 : tensor<256x8xf32>
          %3144 = stablehlo.log_plus_one %3143 : tensor<256x8xf32>
          %3145 = stablehlo.broadcast_in_dim %3118, dims = [0] : (tensor<256xf32>) -> tensor<256x1xf32>
          %3146 = stablehlo.broadcast_in_dim %3145, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x8xf32>
          %3147 = stablehlo.multiply %3146, %3094 : tensor<256x8xf32>
          %3148 = stablehlo.log_plus_one %3147 : tensor<256x8xf32>
          %3149 = stablehlo.subtract %3144, %3148 : tensor<256x8xf32>
          %3150 = stablehlo.multiply %3140, %3149 : tensor<256x8xf32>
          %3151 = stablehlo.subtract %3135, %3150 : tensor<256x8xf32>
          %cst_377 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
          %3152 = stablehlo.broadcast_in_dim %cst_377, dims = [] : (tensor<f32>) -> tensor<256x8xf32>
          %3153 = stablehlo.multiply %3152, %3151 : tensor<256x8xf32>
          %cst_378 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3154 = stablehlo.broadcast_in_dim %cst_378, dims = [] : (tensor<f32>) -> tensor<256xf32>
          %3155 = stablehlo.compare GT, %iterArg_358, %3154, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
          %3156 = func.call @_where_5(%3155, %3153, %3115) : (tensor<256xi1>, tensor<256x8xf32>, tensor<256x8xf32>) -> tensor<256x8xf32>
          %3157 = stablehlo.multiply %iterArg_359, %3156 : tensor<256x8xf32>
          %cst_379 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
          %3158 = stablehlo.reduce(%3157 init: %cst_379) applies stablehlo.add across dimensions = [1] : (tensor<256x8xf32>, tensor<f32>) -> tensor<256xf32>
          %c_380 = stablehlo.constant dense<1> : tensor<i64>
          %3159 = stablehlo.broadcast_in_dim %c_380, dims = [] : (tensor<i64>) -> tensor<256xi64>
          %3160 = stablehlo.add %iterArg_362, %3159 : tensor<256xi64>
          %3161 = stablehlo.is_finite %iterArg_361 : (tensor<256xf32>) -> tensor<256xi1>
          %3162 = stablehlo.not %3161 : tensor<256xi1>
          %cst_381 = stablehlo.constant dense<1.00011921> : tensor<f32>
          %3163 = stablehlo.broadcast_in_dim %cst_381, dims = [] : (tensor<f32>) -> tensor<256xf32>
          %3164 = stablehlo.multiply %iterArg_353, %3163 : tensor<256xf32>
          %cst_382 = stablehlo.constant dense<1.1920929E-4> : tensor<f32>
          %3165 = stablehlo.broadcast_in_dim %cst_382, dims = [] : (tensor<f32>) -> tensor<256xf32>
          %3166 = stablehlo.add %3164, %3165 : tensor<256xf32>
          %3167 = stablehlo.compare GT, %iterArg_361, %3166, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
          %3168 = stablehlo.or %3162, %3167 : tensor<256xi1>
          %c_383 = stablehlo.constant dense<30> : tensor<i64>
          %3169 = stablehlo.broadcast_in_dim %c_383, dims = [] : (tensor<i64>) -> tensor<256xi64>
          %3170 = stablehlo.compare LT, %iterArg_362, %3169, SIGNED : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
          %c_384 = stablehlo.constant dense<false> : tensor<i1>
          %3171 = stablehlo.broadcast_in_dim %c_384, dims = [] : (tensor<i1>) -> tensor<256xi1>
          %3172 = stablehlo.compare NE, %3170, %3171, UNSIGNED : (tensor<256xi1>, tensor<256xi1>) -> tensor<256xi1>
          %3173 = stablehlo.convert %3172 : tensor<256xi1>
          %3174 = stablehlo.and %3168, %3173 : tensor<256xi1>
          %3175 = stablehlo.broadcast_in_dim %3174, dims = [0] : (tensor<256xi1>) -> tensor<256x12xi1>
          %3176 = stablehlo.select %3175, %3088, %iterArg_360 : tensor<256x12xi1>, tensor<256x12xf32>
          %3177 = stablehlo.broadcast_in_dim %3174, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
          %3178 = stablehlo.select %3177, %3158, %iterArg_361 : tensor<256xi1>, tensor<256xf32>
          %3179 = stablehlo.broadcast_in_dim %3174, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
          %3180 = stablehlo.select %3179, %3160, %iterArg_362 : tensor<256xi1>, tensor<256xi64>
          stablehlo.return %iterArg_353, %iterArg_354, %iterArg_355, %iterArg_356, %iterArg_357, %iterArg_358, %iterArg_359, %3176, %3178, %3180 : tensor<256xf32>, tensor<256x12xf32>, tensor<8x12xf32>, tensor<256x8xf32>, tensor<256x8xf32>, tensor<256xf32>, tensor<256x8xf32>, tensor<256x12xf32>, tensor<256xf32>, tensor<256xi64>
        }
        %3047 = stablehlo.subtract %iterArg_201, %3046#8 : tensor<256xf32>
        %3048 = stablehlo.abs %3047 : tensor<256xf32>
        %3049 = stablehlo.abs %3046#8 : tensor<256xf32>
        %cst_343 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
        %3050 = stablehlo.broadcast_in_dim %cst_343, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %3051 = stablehlo.add %3049, %3050 : tensor<256xf32>
        %3052 = stablehlo.divide %3048, %3051 : tensor<256xf32>
        %cst_344 = stablehlo.constant dense<1.1920929E-4> : tensor<f32>
        %3053 = stablehlo.broadcast_in_dim %cst_344, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %3054 = stablehlo.compare LT, %3052, %3053, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
        %3055 = stablehlo.subtract %3046#7, %iterArg_200 : tensor<256x12xf32>
        %3056 = stablehlo.abs %3055 : tensor<256x12xf32>
        %cst_345 = stablehlo.constant dense<0xFF800000> : tensor<f32>
        %3057 = stablehlo.reduce(%3056 init: %cst_345) applies stablehlo.maximum across dimensions = [1] : (tensor<256x12xf32>, tensor<f32>) -> tensor<256xf32>
        %3058 = stablehlo.abs %3046#7 : tensor<256x12xf32>
        %cst_346 = stablehlo.constant dense<0xFF800000> : tensor<f32>
        %3059 = stablehlo.reduce(%3058 init: %cst_346) applies stablehlo.maximum across dimensions = [1] : (tensor<256x12xf32>, tensor<f32>) -> tensor<256xf32>
        %cst_347 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %3060 = stablehlo.broadcast_in_dim %cst_347, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %3061 = stablehlo.add %3060, %3059 : tensor<256xf32>
        %cst_348 = stablehlo.constant dense<1.1920929E-4> : tensor<f32>
        %3062 = stablehlo.broadcast_in_dim %cst_348, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %3063 = stablehlo.multiply %3062, %3061 : tensor<256xf32>
        %3064 = stablehlo.compare LT, %3057, %3063, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
        %cst_349 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
        %3065 = stablehlo.broadcast_in_dim %cst_349, dims = [] : (tensor<f32>) -> tensor<256xf32>
        %3066 = stablehlo.compare LT, %3046#8, %3065, FLOAT : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xi1>
        %3067 = stablehlo.or %3064, %3066 : tensor<256xi1>
        %3068 = stablehlo.and %3054, %3067 : tensor<256xi1>
        %c_350 = stablehlo.constant dense<1> : tensor<i64>
        %3069 = stablehlo.broadcast_in_dim %c_350, dims = [] : (tensor<i64>) -> tensor<256xi64>
        %3070 = stablehlo.add %iterArg_202, %3069 : tensor<256xi64>
        %c_351 = stablehlo.constant dense<50> : tensor<i64>
        %3071 = stablehlo.broadcast_in_dim %c_351, dims = [] : (tensor<i64>) -> tensor<256xi64>
        %3072 = stablehlo.compare LT, %iterArg_202, %3071, SIGNED : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
        %3073 = stablehlo.not %iterArg_203 : tensor<256xi1>
        %c_352 = stablehlo.constant dense<false> : tensor<i1>
        %3074 = stablehlo.broadcast_in_dim %c_352, dims = [] : (tensor<i1>) -> tensor<256xi1>
        %3075 = stablehlo.compare NE, %3072, %3074, UNSIGNED : (tensor<256xi1>, tensor<256xi1>) -> tensor<256xi1>
        %3076 = stablehlo.convert %3075 : tensor<256xi1>
        %3077 = stablehlo.and %3076, %3073 : tensor<256xi1>
        %3078 = stablehlo.broadcast_in_dim %3077, dims = [0] : (tensor<256xi1>) -> tensor<256x12xi1>
        %3079 = stablehlo.select %3078, %3046#7, %iterArg_200 : tensor<256x12xi1>, tensor<256x12xf32>
        %3080 = stablehlo.broadcast_in_dim %3077, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
        %3081 = stablehlo.select %3080, %3046#8, %iterArg_201 : tensor<256xi1>, tensor<256xf32>
        %3082 = stablehlo.broadcast_in_dim %3077, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
        %3083 = stablehlo.select %3082, %3070, %iterArg_202 : tensor<256xi1>, tensor<256xi64>
        %3084 = stablehlo.broadcast_in_dim %3077, dims = [0] : (tensor<256xi1>) -> tensor<256xi1>
        %3085 = stablehlo.select %3084, %3068, %iterArg_203 : tensor<256xi1>, tensor<256xi1>
        stablehlo.return %iterArg_187, %iterArg_188, %iterArg_189, %iterArg_190, %iterArg_191, %iterArg_192, %iterArg_193, %iterArg_194, %iterArg_195, %iterArg_196, %iterArg_197, %iterArg_198, %iterArg_199, %3079, %3081, %3083, %3085 : tensor<8x12xf32>, tensor<256x8xf32>, tensor<256xf32>, tensor<256x8xf32>, tensor<256x8xf32>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<256x12xf32>, tensor<256xf32>, tensor<256xi64>, tensor<256xi1>
      }
      %1517 = func.call @dynamic_update_index_in_dim(%iterArg_49, %1516#13, %iterArg_48) : (tensor<1x256x12xf32>, tensor<256x12xf32>, tensor<i64>) -> tensor<1x256x12xf32>
      %1518 = func.call @dynamic_update_index_in_dim_7(%iterArg_50, %1516#14, %iterArg_48) : (tensor<1x256xf32>, tensor<256xf32>, tensor<i64>) -> tensor<1x256xf32>
      %1519 = func.call @dynamic_update_index_in_dim_8(%iterArg_51, %1516#15, %iterArg_48) : (tensor<1x256xi64>, tensor<256xi64>, tensor<i64>) -> tensor<1x256xi64>
      %1520 = func.call @dynamic_update_index_in_dim_9(%iterArg_52, %1516#16, %iterArg_48) : (tensor<1x256xi1>, tensor<256xi1>, tensor<i64>) -> tensor<1x256xi1>
      %c_186 = stablehlo.constant dense<1> : tensor<i64>
      %1521 = stablehlo.add %iterArg_48, %c_186 : tensor<i64>
      stablehlo.return %iterArg, %iterArg_27, %iterArg_28, %iterArg_29, %iterArg_30, %iterArg_31, %iterArg_32, %iterArg_33, %iterArg_34, %iterArg_35, %iterArg_36, %iterArg_37, %iterArg_38, %iterArg_39, %iterArg_40, %iterArg_41, %iterArg_42, %iterArg_43, %iterArg_44, %iterArg_45, %iterArg_46, %iterArg_47, %1521, %1517, %1518, %1519, %1520 : tensor<1x256x8xf32>, tensor<1x256x8xf32>, tensor<1x256x8xf32>, tensor<1x256xf32>, tensor<1x256x12xf32>, tensor<8x12xf32>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi64>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<78xi1>, tensor<i64>, tensor<1x256x12xf32>, tensor<1x256xf32>, tensor<1x256xi64>, tensor<1x256xi1>
    }
    %10 = stablehlo.reshape %9#23 : (tensor<1x256x12xf32>) -> tensor<256x12xf32>
    %11 = stablehlo.reshape %9#24 : (tensor<1x256xf32>) -> tensor<256xf32>
    %12 = stablehlo.reshape %9#25 : (tensor<1x256xi64>) -> tensor<256xi64>
    %13 = stablehlo.reshape %9#26 : (tensor<1x256xi1>) -> tensor<256xi1>
    return %10, %11, %12, %13 : tensor<256x12xf32>, tensor<256xf32>, tensor<256xi64>, tensor<256xi1>
  }
  func.func private @dynamic_index_in_dim(%arg0: tensor<1x256x8xf32>, %arg1: tensor<i64>) -> tensor<256x8xf32> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c_0, sizes = [1, 256, 8] : (tensor<1x256x8xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x8xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x256x8xf32>) -> tensor<256x8xf32>
    return %1 : tensor<256x8xf32>
  }
  func.func private @dynamic_index_in_dim_0(%arg0: tensor<1x256xf32>, %arg1: tensor<i64>) -> tensor<256xf32> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, sizes = [1, 256] : (tensor<1x256xf32>, tensor<i64>, tensor<i64>) -> tensor<1x256xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x256xf32>) -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @dynamic_index_in_dim_1(%arg0: tensor<1x256x12xf32>, %arg1: tensor<i64>) -> tensor<256x12xf32> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c_0, sizes = [1, 256, 12] : (tensor<1x256x12xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x12xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x256x12xf32>) -> tensor<256x12xf32>
    return %1 : tensor<256x12xf32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<256x8xf32>, %arg2: tensor<256x8xf32>) -> tensor<256x8xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<256x8xf32>
    return %0 : tensor<256x8xf32>
  }
  func.func private @_where_2(%arg0: tensor<i1>, %arg1: tensor<256x12xf32>, %arg2: tensor<12x256xf32>) -> tensor<256x12xf32> {
    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<12x256xf32>) -> tensor<256x12xf32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<i1>, tensor<256x12xf32>
    return %1 : tensor<256x12xf32>
  }
  func.func private @_where_3(%arg0: tensor<256x8xi1>, %arg1: tensor<256x8xf32>, %arg2: tensor<8xf32>) -> tensor<256x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<8xf32>) -> tensor<256x8xf32>
    %1 = stablehlo.select %arg0, %arg1, %0 : tensor<256x8xi1>, tensor<256x8xf32>
    return %1 : tensor<256x8xf32>
  }
  func.func private @_where_4(%arg0: tensor<256xi1>, %arg1: tensor<256xf32>, %arg2: tensor<f64>) -> tensor<256xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<f64>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<256xi1>, tensor<256xf32>
    return %2 : tensor<256xf32>
  }
  func.func private @_where_5(%arg0: tensor<256xi1>, %arg1: tensor<256x8xf32>, %arg2: tensor<256x8xf32>) -> tensor<256x8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<256xi1>) -> tensor<256x8xi1>
    %1 = stablehlo.select %0, %arg1, %arg2 : tensor<256x8xi1>, tensor<256x8xf32>
    return %1 : tensor<256x8xf32>
  }
  func.func private @_where_6(%arg0: tensor<256x8xi1>, %arg1: tensor<256x8xf32>, %arg2: tensor<256x8xf32>) -> tensor<256x8xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<256x8xi1>, tensor<256x8xf32>
    return %0 : tensor<256x8xf32>
  }
  func.func private @dynamic_update_index_in_dim(%arg0: tensor<1x256x12xf32>, %arg1: tensor<256x12xf32>, %arg2: tensor<i64>) -> tensor<1x256x12xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1, 2] : (tensor<256x12xf32>) -> tensor<1x256x12xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c, %c_0 : (tensor<1x256x12xf32>, tensor<1x256x12xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x256x12xf32>
    return %1 : tensor<1x256x12xf32>
  }
  func.func private @dynamic_update_index_in_dim_7(%arg0: tensor<1x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<i64>) -> tensor<1x256xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %c : (tensor<1x256xf32>, tensor<1x256xf32>, tensor<i64>, tensor<i64>) -> tensor<1x256xf32>
    return %1 : tensor<1x256xf32>
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
