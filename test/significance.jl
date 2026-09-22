@testset "Significance adjustments" begin
    function localmoran_with_pvalues(p)
        n = length(p)
        LocalMoran(n, zeros(n), collect(p), zeros(0, 0), zeros(n), zeros(n), zeros(n), fill(:HH, n))
    end

    @testset "Benjamini-Hochberg FDR" begin
        # A failed early rank does not prevent later ranks from passing.
        x = localmoran_with_pvalues([0.02, 0.021, 0.049])
        @test issignificant(x, 0.05, adjust = :fdr) == [true, true, true]

        @test issignificant(localmoran_with_pvalues([0.01, 0.02]), 0.05, adjust = :fdr) == [true, true]
        @test issignificant(localmoran_with_pvalues([0.03, 0.2]), 0.05, adjust = :fdr) == [false, false]

        # The BH cutoff is inclusive, and output follows the original order.
        p = [0.2, 0.02, 0.02]
        x = localmoran_with_pvalues(p)
        p_snapshot = copy(pvalue(x))
        @test issignificant(x, 0.05, adjust = :fdr) == [false, true, true]
        @test pvalue(x) == p_snapshot
        @test p == [0.2, 0.02, 0.02]
        @test issignificant(localmoran_with_pvalues([0.01, 0.05]), 0.05, adjust = :fdr) == [true, true]
    end

    @testset "Existing strict adjustments" begin
        x = localmoran_with_pvalues([0.01, 0.025])
        @test issignificant(x, 0.05, adjust = :none) == [true, true]
        @test issignificant(x, 0.05, adjust = :bonferroni) == [true, false]
        @test issignificant(localmoran_with_pvalues([0.01, 0.05]), 0.05, adjust = :none) == [true, false]
    end
end
