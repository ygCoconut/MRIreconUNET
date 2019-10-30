function SP_tool(varargin);
%
%

if isempty(varargin) 
  Action = 'New';
else
  Action = varargin{1};  
end

switch Action,
 case 'New',
  Create_New_Button;
  
 case 'Save',
  Save_Prefs;
  
 otherwise
  disp(['Unimplemented Functionality: ', Action]);

end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Create_New_Button
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = gcf;

% Find handle for current image toolbar and menubar
hToolbar = findall(fig, 'type', 'uitoolbar', 'Tag','FigureToolBar' );
hToolMenu = findall(fig, 'Label', '&Tools');

if ~isempty(hToolbar) & isempty(findobj(hToolbar, 'Tag', 'figSavePrefsTool')),
  hToolbar_Children = get(hToolbar, 'Children');
	
% The default button size is 15 x 16 x 3. Create Button Image
  button_size_x= 16;
  button_image = NaN* zeros(15,button_size_x);
    
  f = [ ...
      50    51    52    53    54    55    56    57    65    68 ...
      72    80    84    87    95    100   102   110   116   117 ...
      125   130   132   140   144   147   155   156   157   158 ...
      159   160   161   162   172   186   200   214   228 ...
      ];
   
  button_image(f) = 0;
  button_image = repmat(button_image, [1,1,3]);
   
  buttontags = {'figWindowLevel', 'figPanZoom', 'figROITool', 'figViewImages', 'figPointTool', 'figRotateTool', 'figProfileTool','figPointTool'};
  separator = 'off';
   
  hbuttons = [];
  for i = 1:length(buttontags)
    hbuttons = [hbuttons, findobj(hToolbar_Children, 'Tag', buttontags{i})];
  end;
  if isempty(hbuttons)
    separator = 'on';
  end;
   
  hNewButton = uipushtool(hToolbar);
  set(hNewButton, 'Cdata', button_image, ...
                  'Clicked', 'SP_tool(''Save'')',...
                  'Tag', 'figSavePrefsTool', ...
                  'TooltipString', 'Save Preferences Tool',...
                  'Separator', separator, ...
                  'UserData', [], ...
                  'Enable', 'off');   
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Save_Prefs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
hNewButton = gcbo;
udata = get(hNewButton,'Userdata');
h_fig = udata{1};
fig_file = udata{2};
optional_uicontrols = udata{3};

% assume name of structure is hgS_050200
load(fig_file,'-mat');

handles = guihandles(h_fig);
for i=1:length(hgS_050200.children),
	if ~isempty(hgS_050200.children(i).properties) ...
			& isfield(hgS_050200.children(i).properties,'Tag'),
		for j=1:size(optional_uicontrols,1),
			if strcmpi(hgS_050200.children(i).properties.Tag,	optional_uicontrols{j,1}),
				hgS_050200.children(i).properties = ...
					setfield(hgS_050200.children(i).properties, ...
					optional_uicontrols{j,2} , ...
					get(getfield(handles,optional_uicontrols{j,1}), ...
					optional_uicontrols{j,2}));		
				
				% dynamic field setting not supported in earlier versions
				%         hgS_050200.children(i).properties.(optional_uicontrols{j,2}) = ...
				%             get(getfield(handles,optional_uicontrols{j,1}), ...
				%                 optional_uicontrols{j,2});	
				break;
			end;
		end;
	end;
end;
save(which(fig_file),'hgS_050200','-mat');
