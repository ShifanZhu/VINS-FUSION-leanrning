/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "marginalization_factor.h"


void ResidualBlockInfo::Evaluate()
{
    // cost_function就是ceres用的CostFunction，在其中我们重载的Evaluate函数实现了1.计算残差2.计算雅克比
    residuals.resize(cost_function->num_residuals()); // 确定残差的维数

    std::vector<int> block_sizes = cost_function->parameter_block_sizes(); // 确定相关的参数块数目
    // 1.这里得到的是一个指针数组，所以相当于一个二维数组（但又不完全一样，其中每个指针指向的一维数组长度可以不一样）
    // 2.new这个数组的目的和Eigen::Map的作用类似，就是建立ceres优化用的double数组和维护的Eigen数据类型
    //   之间的内存关系，这样优化之后就相当于直接把优化结果存到了Eigen数据类型中，节省了新建中间变量的操作
    raw_jacobians = new double *[block_sizes.size()]; // ceres接口都是double数组，因此这里给雅克比准备数组
    jacobians.resize(block_sizes.size());

    // 这里就是把jacobians每个matrix地址赋给raw_jacobians，然后把raw_jacobians传递给ceres的接口，这样计算结果直接放进了这个matrix
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        // 每个参数块对应的雅克比矩阵的维度，都是 残差维度x变量维度，比如IMU残差维度就是15，视觉残差维度就是2
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]); // 雅克比矩阵大小 残差×变量
        // 建立内存之间的关系。注意：.data()是返回这段数据的首地址，也就是指针，而不是返回这段数据
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    // 调用各自重载的接口计算残差和雅克比
    // 这里需要手动调用ceres的函数计算一下雅克比，传入的形参：参数块二维数组的地址，残差数组的地址，计算的雅克比数组的地址
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    // 如果有核函数，那么就对残差进行相关调整
    //! 问题：对核函数的这个调整没有特别明白？
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm(); // 获得残差的模
        loss_function->Evaluate(sq_norm, rho); // rho[0]:核函数这个点的值 rho[1]这个点的导数 rho[2]这个点的二阶导数
        // printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        // 核函数这点的导数值开根号
        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) // 柯西核p = log(s+1),rho[2]<＝0始终成立，一般核函数二阶导数都是小于0
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else // 这个else一般就不会进入了
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        // 这里就相当于残差雅克比都乘上sqrt_rho1_，及核函数所在的点的一阶导，基本都是小于1
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            // alpha_sq_norm_都是0，不用管了
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}

// 将定义的 residual_block_info 添加到 marginalization_info
// 这里其实就是分别将不同损失函数对应的优化变量、边缘化位置存入到 parameter_block_sizes 和 parameter_block_idx 中，
// 这里注意的是执行到这一步， parameter_block_idx 中仅仅有待边缘化的优化变量的内存地址的key，而且其对应value全部为0
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info); // 残差块收集起来

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;//parameter_blocks里面放的是marg相关的变量
    // parameter_block_sizes是ceres中的一个接口，可以得到各个参数块的大小。
    // 因为本函数传入的是参数块的首地址，并不知道参数块的大小
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes(); // 各个参数块的大小

    // 遍历这些参数块，对于IMU预积分的约束来说，就是4个参数块
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)//这里应该是优化的变量
    {
        double *addr = parameter_blocks[i]; // 指向数据的指针 当前这个参数块的首地址
        int size = parameter_block_sizes[i];//因为仅仅有地址不行，还需要有地址指向的这个数据的长度
        parameter_block_size[reinterpret_cast<long>(addr)] = size;//将指针强转为数据的地址
    }

    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)//这里应该是待边缘化的变量
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];//这个是待边缘化的变量的id
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;//将需要marg的变量的id存入parameter_block_idx
    }
}


// 之前通过调用 addResidualBlockInfo() 已经确定marg变量的数量、存储位置、长度以及待优化变量的数量以及存储位置，下面就需要调用 preMarginalize() 进行预处理
void MarginalizationInfo::preMarginalize()
{
    for (auto it : factors)//在前面的 addResidualBlockInfo 中会将不同的残差块加入到factor中
    {
        // 手动调用ResidualBlockInfo::Evaluate()函数计算每个残差块的残差和雅克比，其内部本质上还是调用了我们定义的各种Factor因子里面的Evaluate函数，
        // 比如IMU预积分因子、视觉重投影因子，我们在这些因子中的Evaluate函数中定义了残差和雅克比如何计算，从而提供给ceres进行非线性优化使用
        it->Evaluate();//利用c++的多态性分别计算所有状态变量构成的残差和雅克比矩阵,其实就是调用各个损失函数中的重载函数Evaluate()

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);//优化变量的地址
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end())//parameter_block_data是整个优化变量的数据
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);//重新开辟一块内存
                parameter_block_data[addr] = data;//通过之前的优化变量的数据的地址和新开辟的内存数据进行关联
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for (auto it : p->sub_factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}


// 正式开始进行边缘化操作
void MarginalizationInfo::marginalize()
{
    /* -------------------会先补充parameter_block_idx，前面提到经过addResidualBlockInfo()函数仅仅带边缘化的优化变量在parameter_block_idx有key值，
    这里会将保留的优化变量的内存地址作为key值补充进去，并统一他们的value值是其前面已经放入parameter_block_idx的优化变量的维度之和，同时这里会计算出两个变量m和n，
    他们分别是待边缘化的优化变量的维度和以及保留的优化变量的维度和。*/
    int pos = 0; // 在所有参数X中排序的idx，注意为了方便舒尔补操作，需要待边缘化的排在前面
    // 下面这个for循环就是为了得到待边缘化的参数块在整个X中的位置，注意待边缘化的参数是排在前面的
    // parameter_block_idx : key是各个待边缘化参数块地址 value预设都是0
    for (auto &it : parameter_block_idx)//遍历待marg的优化变量的内存地址
    {
        it.second = pos; // 这就是在所有参数中排序的idx，待边缘化的排在前面
        // 因为要进行求导，因此大小时local size，具体一点就是使用李代数
        // 因为H矩阵最终是为了求δx，所以要使用李代数（或者说旋转向量），使用四元数会过参数化
        pos += localSize(parameter_block_size[it.first]); // localSize只是处理位姿，7变6
    }

    m = pos; // 总共待边缘化的参数块总大小（不是个数）
    
    // 其他参数块，也就是不被边缘化的哪些参数块
    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())//如果这个变量不是是待marg的优化变量
        {
            // 可见这里对parameter_block_idx这个map进行了扩充，原来存储的只是被边缘化的哪些参数块
            // 现在在后面追加了不被边缘化的哪些参数块。也就是通过上述操作，把边缘化的参数块位置调整到了最前面
            parameter_block_idx[it.first] = pos; // 这样每个参数块的大小都能被正确找到
            pos += localSize(it.second);//pos加上这个变量的长度
        }
    }

    // 注意这里出了上面的for循环后，pos被累加成了整个H矩阵的维度大小，包括边缘化的部分的大小
    n = pos - m;//要保留下来的变量个数
    //ROS_INFO("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());
    if(m == 0)
    {
        valid = false;
        printf("unstable tracking...\n");
        return;
    }


    //通过上面的操作就会将所有的优化变量进行一个伪排序，待marg的优化变量的idx为0，其他的和起所在的位置相关
    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos); // Ax = b预设大小
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread

    // 往A矩阵和b矩阵中填东西，利用多线程加速
    // 函数会通过多线程快速构造各个残差对应的各个优化变量的信息矩阵（雅克比和残差前面都已经求出来了），然后在加起来.
    // 因为这里构造信息矩阵时采用的正是parameter_block_idx作为构造顺序，因此，就会自然而然地将待边缘化的变量构造在矩阵的左上方。
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS]; // 开了4个线程，假设有20个CostFunction，那么每个线程计算5个CostFunction
    ThreadsStruct threadsstruct[NUM_THREADS]; // 多线程执行函数的入口参数
    int i = 0;
    // factors是一个数组，其中每个元素是ResidualBlockInfo类的指针
    for (auto it : factors)//将各个残差块的雅克比矩阵分配到各个线程中去
    {
        threadsstruct[i].sub_factors.push_back(it); // 每个线程均匀分配任务
        i++;
        i = i % NUM_THREADS;
    }
    // 每个线程构造一个A矩阵和b矩阵，最后大家加起来
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        // 所以A矩阵和b矩阵大小一样，预设都是0
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        // 多线程访问会带来冲突，因此每个线程备份一下要查询的map
        threadsstruct[i].parameter_block_size = parameter_block_size; // 大小
        threadsstruct[i].parameter_block_idx = parameter_block_idx;   // 索引
        // 产生若干线程
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));//分别构造矩阵
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        // 等待各个线程完成各自的任务
        pthread_join( tids[i], NULL );
        // 把各个子模块拼起来，就是最终的Hx = g的矩阵了
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());

    // 注意：到这里就得到了Hx=b中的H和b矩阵，这个时候如果直接求解出来x，就相当于实现了一个手写后端的功能
    // ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    // ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());

    //  Step  下面开始进行舒尔补实现边缘化
    //  Amm矩阵的构建是为了保证其正定性
    // 这里的Amm就是Haa，也就是H矩阵的左上角。这里保证正定性是为了求逆，因为如果不正定的话，Haa无法求逆
    //? 问题：为什么可以这样做呢？这样处理之后和原来的矩阵相差多少？

    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm); // 特征值分解，为了更快的进行矩阵求逆

    // ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());
    //  一个逆矩阵的特征值是原矩阵的倒数，特征向量相同　select类似c++中 ? :运算符
    //  利用特征值取逆来构造其逆矩阵
    //! 问题： 1.利用特征值分解求逆矩阵：(下面的推导还是有点问题，主要是B^-1这部分；另外和代码对不上)
    //   Ax = yx，其中A是待分解的矩阵，x是特征向量，y是特征值
    //   那么所有特征值和特征向量可以写成矩阵的形式：
    //      A[x1, x2, ... , xn] = diag(y1, y2, ... , yn)[x1, x2, ... , xn]
    //   记 B = [x1, x2, ... , xn]，则 A = B^-1 * diag(yn) * B
    //   所以A^-1 = B^-1 * diag(1/yn) * B
    // 2. Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0))
    //      意思就是判断特征值是否>无穷小（实际就是判断是否>0），如果不是就取0，然后取逆（也就是每个特征值取倒数）
    //    最后的asDiagonal()就是生成一个对角矩阵
    
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    // -----------函数会通过shur补操作进行边缘化，然后再从边缘化后的信息矩阵中恢复出来雅克比矩阵 linearized_jacobians 和残差 linearized_residuals ，
    //------------这两者会作为先验残差带入到下一轮的先验残差的雅克比和残差的计算当中去。
    // 舒尔补
    // 假设Ax = b，其中x = [xa,xb]'，其中xa是要被边缘化的部分, 则Ax = b被划分如下：
    //     [Amm Amr][xa] = [bmm]
    //     [Arm Arr][xb] = [brr]
    // 则进行舒尔补消掉Arm，最后化成：(Arr - Arm*Amm^-1*Amr)xb = brr - Arm*Amm^-1*bmm
    Eigen::VectorXd bmm = b.segment(0, m); // 带边缘化的大小，Eigen::Vector.segment(i,n):向量从i开始的n个元素
    Eigen::MatrixXd Amr = A.block(0, m, m, n); // 对应的四块矩阵
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n); // 剩下的参数
    A = Arr - Arm * Amm_inv * Amr;//这里的A和b应该都是marg过的A和b,大小是发生了变化的
    b = brr - Arm * Amm_inv * bmm;
    // ----------------   舒尔补操作完毕  -----------------

    // 下面就是更新先验残差项
    //  Step 下面对舒尔补之后的Hx = b，进行分解得到 (J^T*J) * x = -J^t*e 中的残差和雅克比
    //  这个地方根据Ax = b => JT*J = - JT * e
    //  对A做特征值分解 A = V * S * VT,其中Ｓ是特征值构成的对角矩阵，V是特征向量构成的矩阵，注意其逆等于其转置
    //  所以J = S^(1/2) * VT ,
    //  这样 JT * J = (S^(1/2) * VT)T * S^(1/2) * VT
    //            = V * S^(1/2)T *  S^(1/2) * VT = V * S * VT(对角矩阵的转置等于其本身)
    //  e = -(JT)-1 * b = - (S^-(1/2) * V^-1) * b = - (S^-(1/2) * VT) * b(V的转置等于其逆)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);//求特征值
    // 构造S矩阵和S的逆矩阵
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0)); // numerical stability
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));
    // 对S矩阵和其逆矩阵进行开根号，因为是对角矩阵，所以开根号很容易
    Eigen::VectorXd S_sqrt = S.cwiseSqrt(); // element-wise square root
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    // 重要：解释了为什么把边缘化得到的Hx=b分解成(J^T*J) * x = -J^t*e的形式：
    // 边缘化为了实现对剩下参数块的约束，为了便于一起优化，就抽象成了残差和雅克比的形式，这样也形成了一种残差约束
    // linearized_jacobians is directly related to A through its eigenvalues and eigenvectors, and it encodes important info about the variance and directions in the data that A represents.
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

// 先验残差的损失函数
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);
};

/* Evaluate 计算所有状态变量构成的残差和雅克比矩阵
这个函数通过传入的优化变量值parameters，以及先验值（对于先验残差就是上一时刻的先验残差last_marginalization_info，
对于IMU就是预计分值pre_integrations[1]，对于视觉就是空间的的像素坐标pts_i, pts_j）
可以计算出各项残差值residuals，以及残差对应优化变量的雅克比矩阵jacobians。
参考:https://blog.csdn.net/weixin_44580210/article/details/95748091 */
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    int n = marginalization_info->n; // 上一次边缘化保留的残差块的local size的和,也就是残差维数
    int m = marginalization_info->m; // 上次边缘化的被margin的残差块总和
    Eigen::VectorXd dx(n);           // 用来存储残差

    // 遍历所有的剩下的有约束的残差块
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i]; // 当前这个参数块的维度
        int idx = marginalization_info->keep_block_idx[i] - m; // idx起点统一到0
        // parameters是本函数传入的形参，也就是这次要更新的参数块
        // creating a read-only map(view) (const) of a block of memory as an Eigen vector
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size); // 当前参数块的值
        //? 问题：当时参数块的值，是指在边缘化操作之前这个参数块的值？ 回答 ： 是的！
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            // 这个dx的含义就是当时边缘化的时候这个参数块的值，到现在经过一些优化了这个参数块的值，增加了多少
            // 比如上次边缘化的时候这个参数块值是5，现在经过一些优化之后这个参数块的值变成了5.2，那么dx就是0.2
            dx.segment(idx, size) = x - x0; // 不需要local param的直接做差
        else // 代表位姿的param
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>(); // 位移直接做差
            // 旋转就是李代数做差，这里四元数相乘做差取虚部在x2，结果就是旋转向量的部分
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            // 确保实部大于0，如果不大于0那么就把旋转向量部分取个反
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // 更新残差　边缘化后的先验误差 e = e0 + J * dx
    // 个人理解：根据FEJ（First Estimated Jacobian）．雅克比保持不变，但是残差随着优化会变化，
    //         因此下面不更新雅克比　只更新残差
    // 可以参考　https://blog.csdn.net/weixin_41394379/article/details/89975386
    // 这里的残差就体现了边缘化产生的先验如何对当前滑窗中的参数块产生约束：
    //   如果dx比较大，也就是上次边缘化之前和现在参数块的值变化比较大，那这里的残差就会比较大，这样就会让
    //   最后总的loss比较大，所以优化的时候就需要考虑调整这些参数块的值，让边缘化约束产生的残差减小
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {
        // 遍历每一个参数块
        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            // 由于雅克比是固定的，所以只要从上次边缘化求出来的总的雅克比矩阵中取出来当前参数块对应的雅克比即可
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                // 和误差同理，索引要对齐到0。但是注意这里是为了求列索引，因为边缘化的Jacobian已经确定了是nxn的，其中n是保留下的状态变量的维数
                // 它的分布如下： [d{e}/d{x_m+1}, d{e}/d{x_m+2} ... d{e}/d{x_m+n}]，所以列坐标就是状态变量的索引
                int idx = marginalization_info->keep_block_idx[i] - m;
                // 这部分雅克比的维度是n行，size列
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                // 从上次边缘化求得的总的雅克比矩阵中的第idx列开始，取size列，就是当前参数块对应的雅克比
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
