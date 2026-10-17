function outJson = rlstudio_convert(mdlPath, outPath, specJson, initPath)
%RLSTUDIO_CONVERT Enregistre une COPIE d'un modele Simulink adaptee a RL Studio (le fichier d'origine n'est pas modifie).
%
%   spec (JSON) :
%     inports   : [{path, name, default}]  entrees racine remplacees par une constante pilotable `Rl_<name>`
%     remove    : [path]                   blocs a supprimer (scopes, affichages...) ; leurs sources sont terminees
%     variables : [nom]                    variables du workspace de base copiees dans le workspace du modele
%     observations : [{name, path, port}]  signaux a observer ; une sortie (Outport) est remplacee par le bloc qui l'alimente
%     solver    : 'keep' | 'fixed'         'fixed' impose un solveur ode4 a pas fixe de `solver_step` secondes
%     solver_step : pas du solveur [s]
%     silence_loops : true                 masque l'avertissement de boucle algebrique
%   Le modele converti garde le nom du modele d'origine : les chemins de blocs restent valides.
%   Renvoie un JSON {actions: [{name, block, param}], observations: [{name, block, port}], warnings: [...]}.
if nargin < 4, initPath = ''; end
spec = jsondecode(specJson);
run_init(initPath);
[~, mdl] = fileparts(mdlPath);
if bdIsLoaded(mdl), close_system(mdl, 0); end
load_system(mdlPath);
save_system(mdl, outPath);          % « Enregistrer sous » : le fichier d'origine reste intact
warnings = {};
actions = {};

% ---- 1. variables du workspace de base -> workspace du modele (le modele devient autonome)
mws = get_param(mdl, 'ModelWorkspace');
vars = cellify(spec, 'variables');
for k = 1:numel(vars)
    nm = vars{k};
    try
        mws.assignin(nm, evalin('base', nm));
    catch err
        warnings{end+1} = sprintf('Variable %s non copiee : %s', nm, err.message); %#ok<AGROW>
    end
end

% ---- 2. observations : on resout d'abord les sorties (Outport) avant de toucher au modele
obs = cellify(spec, 'observations');
resolved = cell(1, numel(obs));
for k = 1:numel(obs)
    o = obs{k};
    blk = o.path; port = o.port;
    if strcmp(get_param(blk, 'BlockType'), 'Outport')
        pc = get_param(blk, 'PortConnectivity');
        if isempty(pc) || isempty(pc(1).SrcBlock) || pc(1).SrcBlock <= 0
            error('rlstudio:convert', 'La sortie %s n''est reliee a aucun bloc.', o.name);
        end
        blk = getfullname(pc(1).SrcBlock); port = pc(1).SrcPort + 1;
    end
    resolved{k} = struct('name', o.name, 'block', blk, 'port', port);
end

% ---- 3. entrees racine -> constantes pilotables
ins = cellify(spec, 'inports');
for k = 1:numel(ins)
    i = ins{k};
    cname = ['Rl_' i.name];
    p = i.path;
    pos = get_param(p, 'Position');
    parent = get_param(p, 'Parent');
    ph = get_param(p, 'PortHandles');
    dsts = [];
    lh = get_param(ph.Outport, 'Line');
    if lh > 0
        dsts = get_param(lh, 'DstPortHandle');
        delete_line(lh);               % la ligne doit disparaitre avant le bloc, sinon le port de destination reste occupe
    end
    delete_block(p);
    dflt = 0; if isfield(i, 'default') && ~isempty(i.default), dflt = i.default; end
    nb = add_block('simulink/Sources/Constant', [parent '/' cname], 'Value', num2str(dflt, 17), 'Position', pos, ...
                   'MakeNameUnique', 'on');
    nph = get_param(nb, 'PortHandles');
    for d = reshape(dsts(dsts > 0), 1, [])
        add_line(parent, nph.Outport(1), d, 'autorouting', 'on');
    end
    actions{end+1} = struct('name', i.name, 'block', getfullname(nb), 'param', 'Value'); %#ok<AGROW>
end

% ---- 4. blocs a retirer ; les sorties qui les alimentaient sont terminees pour eviter les ports ouverts
rem = cellify(spec, 'remove');
for k = 1:numel(rem)
    b = rem{k};
    if ~getSimulinkBlockHandle(b), continue; end
    pc = get_param(b, 'PortConnectivity');
    srcs = [];
    for j = 1:numel(pc)
        if ~isempty(pc(j).SrcBlock) && pc(j).SrcBlock > 0
            srcs(end+1, :) = [pc(j).SrcBlock, pc(j).SrcPort]; %#ok<AGROW>
        end
    end
    parent = get_param(b, 'Parent');
    delete_block(b);
    for j = 1:size(srcs, 1)
        sph = get_param(srcs(j, 1), 'PortHandles');
        op = sph.Outport(srcs(j, 2) + 1);
        if get_param(op, 'Line') < 0
            pos = get_param(srcs(j, 1), 'Position');
            t = add_block('simulink/Sinks/Terminator', [parent '/Term_rl'], 'MakeNameUnique', 'on', ...
                          'Position', [pos(3) + 40, pos(2), pos(3) + 60, pos(2) + 20]);
            tph = get_param(t, 'PortHandles');
            add_line(parent, op, tph.Inport(1), 'autorouting', 'on');
        end
    end
end

% ---- 5. solveur et journalisation
if isfield(spec, 'solver') && strcmp(spec.solver, 'fixed')
    set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', num2str(spec.solver_step, 17));
end
set_param(mdl, 'SaveOutput', 'off', 'SaveTime', 'off', 'SaveState', 'off');
if isfield(spec, 'silence_loops') && spec.silence_loops
    set_param(mdl, 'AlgebraicLoopMsg', 'none');
end
set_param(mdl, 'Description', ['Copie convertie par RL Studio de ' mdlPath '. Entrees remplacees par des constantes pilotables, ' ...
    'affichages retires, variables embarquees.']);

save_system(mdl, outPath);
close_system(mdl, 0);
res = struct('model', mdl, 'actions', {actions}, 'observations', {resolved}, 'warnings', {warnings});
outJson = jsonencode(res);
end

function c = cellify(s, field)
% Liste JSON -> cellule, que le champ soit absent, vide, un tableau de structs ou une cellule.
c = {};
if ~isfield(s, field) || isempty(s.(field)), return; end
v = s.(field);
if iscell(v), c = v; elseif isstruct(v), c = num2cell(v); elseif ischar(v), c = {v}; else, c = num2cell(v); end
c = reshape(c, 1, []);
end

function run_init(initPath)
if isempty(initPath), return; end
[~, ~, ext] = fileparts(initPath);
if strcmpi(ext, '.mat')
    evalin('base', sprintf('load(''%s'');', strrep(initPath, '''', '''''')));
else
    evalin('base', sprintf('run(''%s'');', strrep(initPath, '''', '''''')));
end
end
