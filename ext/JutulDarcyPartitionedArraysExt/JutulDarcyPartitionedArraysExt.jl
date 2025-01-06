module JutulDarcyPartitionedArraysExt
    using Jutul, JutulDarcy
    using TimerOutputs
    using PrecompileTools
    # Specific dependencies
    using PartitionedArrays, MPI, HYPRE
    using LinearAlgebra
    using SparseArrays

    import Jutul: JutulCase, PArrayBackend, JutulConfig, BoomerAMGPreconditioner
    import Jutul: PArraySimulator, MPISimulator, PArrayExecutor
    import Jutul: DebugPArrayBackend, JuliaPArrayBackend, MPI_PArrayBackend
    import Jutul: partition_distributed, simulate_parray, @tic
    import JutulDarcy:
        reservoir_partition, partitioner_input, apply_cpr_pressure_stage!,
        apply_cpr_smoother!, set_default_cnv_mb!, setup_reservoir_simulator_parray,
        increment_pressure!, set_dp!, correct_residual!, CPRStorage, CPRPreconditioner,
        number_of_components

    timeit_debug_enabled() = Jutul.timeit_debug_enabled()

    include("cpr.jl")

    function setup_reservoir_simulator_parray(
            case::JutulCase,
            backend::PArrayBackend;
            conn = :logtrans,
            np = missing,
            kwarg...
        )
        if ismissing(np)
            np = MPI.Comm_size(MPI.COMM_WORLD)
        end
        np::Int
        N, T, groups = partitioner_input(case.model, case.parameters, conn = conn)
        rmodel = reservoir_model(case.model)
        nc = number_of_cells(rmodel.domain)
        p_num = partition_distributed(N, T, nc = nc, np = np, groups = groups)
        p = reservoir_partition(case.model, p_num)
        return PArraySimulator(case, p; backend = backend, kwarg...)
    end

    function set_default_cnv_mb!(config::JutulConfig, sim::PArraySimulator; kwarg...)
        simulators = sim.storage[:simulators]
        map(simulators, config[:configs]) do sim, cfg
            set_default_cnv_mb!(cfg, sim)
        end
        return config
    end

    function Jutul.parray_preconditioner_apply!(global_out, main_prec::CPRPreconditioner{<:BoomerAMGPreconditioner, <:Any}, R, preconditioners, simulator, arg...)
        global_cell_vector = simulator.storage.distributed_cell_buffer
        global_buf = simulator.storage.distributed_residual_buffer
        A_ps = main_prec.storage.A_ps

        @. global_out = 0.0
        npre = main_prec.npre
        npost = main_prec.npost
        if npre > 0
            apply_cpr_smoother!(global_out, R, global_buf, preconditioners, A_ps, npre)
        end
        @tic "cpr first stage" map(local_values(R), preconditioners, ghost_values(R)) do r, prec, x_g
            @. x_g = 0.0
            apply_cpr_pressure_stage!(prec, prec.storage, r, arg...)
            nothing
        end
        # The following is an unsafe version of this:
        # copy!(global_cell_vector, main_prec.p)
        p_h = main_prec.storage.p
        @assert !isnothing(p_h) "CPR is not properly initialized."
        @tic "hypre GetValues" map(
            own_values(global_cell_vector),
            preconditioners
        ) do ov, prec
            helper = prec.pressure_precond.data[:assembly_helper]
            bz = prec.storage.block_size
            indices = helper.indices
            indices::Vector{HYPRE.HYPRE_BigInt}
            nvalues = indices[end] - indices[1] + 1
            HYPRE.@check HYPRE.HYPRE_IJVectorGetValues(p_h, nvalues, indices, ov)
        end

        @tic "set dp" map(own_values(global_out), own_values(global_cell_vector), preconditioners) do dx, dp, prec
            bz = prec.storage.block_size
            increment_pressure!(dx, dp, bz, prec.storage.p_buffer)
        end
        # End unsafe shenanigans
        if npost > 0
            @tic "set dp" map(
                    own_values(global_buf),
                    own_values(global_cell_vector),
                    preconditioners
                ) do buf, dp, prec
                bz = prec.storage.block_size
                for i in eachindex(dp)
                    set_dp!(buf, bz, dp, i)
                end
            end
            correct_residual!(R, A_ps, global_buf)
            apply_cpr_smoother!(global_out, R, global_buf, preconditioners, A_ps, npost, skip_last = true)
        end
        @tic "communication" consistent!(global_out) |> wait
        return global_out
    end

    function apply_cpr_smoother!(X::PVector, R::PVector, Buf::PVector, prec, A_ps, n; skip_last = false)
        for i in 1:n
            map(
                local_values(Buf),
                local_values(R),
                own_values(X),
                prec,
                ghost_values(R),
                own_values(Buf)
            ) do buf, r, x, p, r_g, buf_o
                @. r_g = 0.0
                apply!(buf, p.system_precond, r)
            end
            @. X += Buf
            if i < n || !skip_last
                correct_residual!(R, A_ps, Buf)
            end
        end
    end

    function Jutul.parray_update_preconditioners!(sim::PArraySimulator, cpr::CPRPreconditioner{<:BoomerAMGPreconditioner, <:Any}, preconditioners, recorder)
        offset = sim.storage.process_offset
        n = sim.storage.nc_process
        comm = sim.storage.comm
        if sim.storage[:number_of_processes] > 1
            @assert sim.backend isa Jutul.MPI_PArrayBackend "Cannot use HYPRE with emulated multiple processes. Backend was $(sim.backend)"
        end

        function create_hypre_vector()
            x = HYPREVector(comm, offset + 1, offset + n)
            asm = HYPRE.start_assemble!(x)
            HYPRE.finish_assemble!(asm)
            return x
        end
        if isnothing(cpr.storage)
            A_p = HYPREMatrix(comm, offset + 1, offset + n)
            r_p = create_hypre_vector()
            p = create_hypre_vector()

            global_sol_buf = sim.storage.distributed_solution_buffer
            global_res_buf = sim.storage.distributed_residual_buffer
            A_ps = Jutul.parray_linear_system_operator(sim.storage.simulators, length(global_res_buf))
            p_sys = (A_p, r_p, p)
            rmodel = reservoir_model(sim.storage.model)
            bz = degrees_of_freedom_per_entity(rmodel, Cells())
            ncomp = number_of_components(rmodel.system)
            cpr.storage = CPRStorage(n, bz, A_ps, p_sys, global_sol_buf, global_res_buf; ncomp = ncomp)
        end
        cpr_storage = cpr.storage
        A_p = cpr_storage.A_p
        A_ps = cpr_storage.A_ps
        r_p = cpr_storage.r_p
        x_p = cpr_storage.p
        bz = cpr_storage.block_size
        w_rhs = cpr_storage.w_rhs
        ncomp = cpr_storage.number_of_components

        map(sim.storage.simulators, preconditioners) do sim, prec
            storage = Jutul.get_simulator_storage(sim)
            model = Jutul.get_simulator_model(sim)
            sys = storage.LinearizedSystem
            ctx = model.context
            rsys = JutulDarcy.reservoir_jacobian(sys)
            rid = objectid(rsys)
            prec.pressure_precond.data[:hypre_system] = (A_p, r_p, x_p)
            if isnothing(prec.storage)
                w_p = zeros(bz, n)
                prec.storage = CPRStorage(A_p, r_p, x_p, missing, missing, A_ps, w_p, w_rhs, n, bz, ncomp, rid, zeros(n), nothing, Float64)
            else
                @assert prec.storage.id == rid
            end
            Jutul.update_preconditioner!(prec, sys, ctx, model, storage, recorder, sim.executor)
            prec
        end
        return (cpr, preconditioners)
    end

    @compile_workload begin
    #     targets = [(true, :csc), (true, :csr)]
    #     # MPI, trivial partition
    #     JutulDarcy.precompile_darcy_multimodels(targets,
    #         dims = (4, 1, 1),
    #         default_linsolve = false,
    #         setuparg = (
    #             mode = :mpi,
    #             precond = :ilu0
    #             ),
    #         split_wells = true
    #     )
    #     # Native PArray, non-trivial partition
    #     JutulDarcy.precompile_darcy_multimodels(targets,
    #         dims = (4, 1, 1),
    #         default_linsolve = false,
    #         setuparg = (
    #             mode = :parray,
    #             parray_arg = (np = 2, ),
    #             precond = :ilu0
    #             ),
    #         split_wells = true
    #     )
    end
end

