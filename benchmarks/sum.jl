function itersum()
    s::Int64 = 0
    for i in 1:100_000_000
        s += i
    end
    return s
end

result = itersum()
exit(result % 256)
