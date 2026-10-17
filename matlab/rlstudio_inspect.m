function outJson = rlstudio_inspect(mdlPath)
%RLSTUDIO_INSPECT Liste les blocs d'un modele : sorties disponibles et parametres numeriques.
%   Sert a l'explorateur de la GUI pour choisir observations (signaux) et actions (parametres).
[~, mdl] = fileparts(mdlPath);
if ~bdIsLoaded(mdl)
    load_system(mdlPath);
end

blks = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Type', 'Block');
items = cell(1, numel(blks));
for k = 1:numel(blks)
    b = blks{k};
    item = struct('path', b, 'type', get_param(b, 'BlockType'), 'n_out', 0, 'params', {{}});
    try
        pc = get_param(b, 'PortConnectivity');
        ph = get_param(b, 'PortHandles');
        item.n_out = numel(ph.Outport);
    catch
    end
    params = {};
    try
        dp = fieldnames(get_param(b, 'DialogParameters'));
        for j = 1:numel(dp)
            v = get_param(b, dp{j});
            if ischar(v) && ~isempty(v) && isfinite(str2double(v))
                params{end+1} = struct('name', dp{j}, 'value', str2double(v)); %#ok<AGROW>
            end
        end
    catch
    end
    item.params = params;
    items{k} = item;
end
outJson = jsonencode(items);
end
