local function is_prime(n)
    if n < 2 then return false end
    local j = 2
    while j * j <= n do
        if n % j == 0 then return false end
        j = j + 1
    end
    return true
end

local count = 0
for i = 2, 100000 do
    if is_prime(i) then count = count + 1 end
end
os.exit(count % 256)
