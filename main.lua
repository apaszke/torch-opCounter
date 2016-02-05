
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('src/opts.lua')
paths.dofile('src/profiler.lua')

-- Load the model
paths.dofile('src/model.lua')

local input = torch.randn(torch.LongStorage(opt.inputSize)):cuda()

print('Counting ops...')
local total, layer_ops = count_ops(model, input)
print('')


-- Compute per layer opt counts
local per_layer = {}
for i, info in pairs(layer_ops) do
    local name = info['name']
    local ops = info['ops']
    if not per_layer[name] then
        per_layer[name] = 0
    end
    per_layer[name] = per_layer[name] + ops
end

-- Print total
for name, count in pairs(per_layer) do
    if count > 0 then
        local tabs = 4 - math.floor((#name + 1) / 8)
        print(string.format('%s:%s%.2e ops', name, string.rep('\t', tabs), count))
    end
end
print(string.format('%sTotal:\t\t\t\t%.4e ops', sys.COLORS.blue, total))
