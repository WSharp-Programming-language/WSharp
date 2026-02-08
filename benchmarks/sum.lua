local sum = 0
for i = 1, 100000000 do
    sum = sum + i
end
os.exit(sum % 256)
