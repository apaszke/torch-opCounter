
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('src/opts.lua')
paths.dofile('src/profiler.lua')

-- Load the model
paths.dofile('src/model.lua')

local input = torch.randn(torch.LongStorage(opt.inputSize)):cuda()

printVerbose('Counting ops...')
local total, layer_ops = count_ops(model, input)
printVerbose('')


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
        printVerbose(string.format('%-32s%.2e ops', name..':', count))
    end
end
print(string.format('%s%-32s%.4e ops', sys.COLORS.blue, 'Total:', total))
