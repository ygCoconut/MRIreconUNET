function WL_tool(varargin);
% function WL_tool(varargin);
% Window - Level tool for adjusting contrast of a set of images
% interactively. Use with imagescn or with imagescn.
%
% Usage: WL_tool;
%
% Author: Daniel Herzka  herzkad@nih.gov
% Laboratory of Cardiac Energetics 
% National Heart, Lung and Blood Institute, NIH, DHHS
% Bethesda, MD 20892
% and 
% Medical Imaging Laboratory
% Department of Biomedical Engineering
% Johns Hopkins University Schoold of Medicine
% Baltimore, MD 21205


if isempty(varargin) 
	Action = 'New';
else
	Action = varargin{1};  
end;


switch Action
	case 'New'
		Create_New_Button;
	case 'Activate_Window_Level'
		Activate_Window_Level(varargin{:});
		
	case 'Deactivate_Window_Level'
		Deactivate_Window_Level(varargin{2:end});
		
	case 'Adjust_On'
		%Entry
		Adjust_On;
	case 'Adjust_Window_Level'
		% Cycle
		Adjust_Window_Level;
	case 'Adjust_Window_Level_For_All'
		% Exit
		set(gcf, 'WindowButtonMotionFcn', ' ');
		Adjust_Window_Level_For_All;
		
	case  'Edit_Adjust'
		Edit_Adjust;
		
	case 'Set_Colormap'
		Set_Colormap(varargin{2:end});
		
	case 'Menu_Window_Level'
		Menu_Window_Level; 
		
	case 'WL_Reset'
		WL_Reset;
		
	case 'Auto_WL_Reset'
		Auto_WL_Reset;
		
	case 'Close_Parent_Figure'
		Close_Parent_Figure;
		
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

% if the toolbar exists and the button has not been previously created
if ~isempty(hToolbar) & isempty(findobj(hToolbar, 'Tag', 'figWindowLevel'))
	hToolbar_Children = get(hToolbar, 'Children');
	% The default button size is 15 x 16 x 3. Create Button Image
	button_size_x= 16;
	button_image = repmat(linspace(0,1,button_size_x), [ 15 1 3]);
	
	buttontags = {'figWindowLevel', 'figPanZoom', 'figROITool', 'figViewImages', 'figPointTool', 'figRotateTool', 'figProfileTool'};
	separator = 'off';
	
	hbuttons = [];
	for i = 1:length(buttontags)
		hbuttons = [hbuttons, findobj(hToolbar_Children, 'Tag', buttontags{i})];
	end;
	if isempty(hbuttons)
		separator = 'on';
	end;
	
	hNewButton = uitoggletool(hToolbar);
	set(hNewButton, 'Cdata', button_image, ...
		'OnCallback', 'WL_tool(''Activate_Window_Level'');',...
		'OffCallback', 'WL_tool(''Deactivate_Window_Level'');',...
		'Separator', 'On',...
		'Tag', 'figWindowLevel', ...
		'TooltipString', 'Set Image Window Level',...
		'Separator', separator, ...
		'UserData', [], ...
		'Enable', 'on');
end;

if ~isempty(hToolMenu) & isempty(findobj(hToolMenu,'Tag', 'menuWindowLevel')) 
	
	hWindowLevelMenu = findobj(hToolMenu, 'Tag', 'menuWindowLevel');
	hPanZoomMenu     = findobj(hToolMenu, 'Tag', 'menuPanZoom');
	hROIToolMenu     = findobj(hToolMenu, 'Tag', 'menuROITool');
	hViewImageMenu   = findobj(hToolMenu, 'Tag', 'menuViewImages');
	hPointToolMenu   = findobj(hToolMenu, 'Tag', 'menuPointTool');
	hRotateToolMenu  = findobj(hToolMenu, 'Tag', 'menuRotateTool');
	hProfileToolMenu = findobj(hToolMenu, 'Tag', 'menuProfileTool');
	  
	position = 9;
	separator = 'On';
	hMenus = [hPanZoomMenu, hROIToolMenu, hViewImageMenu, hPointToolMenu, hRotateToolMenu,hProfileToolMenu];
	
	if length(hMenus>0) 
		position = position + length(hMenus);
		separator = 'Off';
	end;
	
	hNewMenu = uimenu(hToolMenu,'Position', position);    
	set(hNewMenu, 'Tag', 'menuWindowLevel','Label',...
		'Window and Level',...
		'CallBack', 'WL_tool(''Menu_Window_Level'')',...
		'Separator', 'On',...
		'UserData', hNewButton...
		); 
end;
	

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Activate_Window_Level(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
default_colormap = [];
if nargin == 1,
	old_SHH = get(0, 'ShowHiddenHandles');
	set(0, 'ShowHiddenHandles', 'On');
	hNewButton = gcbo;
	set(findobj('Tag', 'menuWindowLevel'),'checked', 'on');
	set(0, 'ShowHiddenHandles', old_SHH);
else
	if ischar(varargin{2}), 
		old_SHH = get(0, 'ShowHiddenHandles');
		set(0, 'ShowHiddenHandles', 'On');
		hNewButton = gcbo;
		set(findobj('Tag', 'menuWindowLevel'),'checked', 'on');
		default_colormap = varargin{2};
		set(0, 'ShowHiddenHandles', old_SHH);
	else
		hNewButton = varargin{2};
	end;
end;

% allows for calls from buttons other than those in toolbarfig = get(hNewButton, 'Parent');
fig = get(hNewButton, 'Parent');
if ~strcmp(get(fig, 'Type'), 'figure'),
	fig = get(fig, 'Parent');
end

% Deactivate zoom and rotate buttons
hToolbar = findall(fig, 'type', 'uitoolbar');
hToolbar = findobj(hToolbar, 'Tag', 'FigureToolBar');

if ~isempty(hToolbar)
	hToolbar_Children = get(hToolbar, 'Children');
	
	% disable MATLAB's own tools
	Rot3D = findobj_hidden(hToolbar_Children,'Tag', 'figToolRotate3D');
	ZoomO = findobj_hidden(hToolbar_Children,'Tag', 'figToolZoomOut');
	ZoomI = findobj_hidden(hToolbar_Children,'Tag', 'figToolZoomIn');

	% try to disable other tools buttons - if they exist
	WL = findobj_hidden(hToolbar_Children, 'Tag', 'figWindowLevel');
	PZ = findobj_hidden(hToolbar_Children, 'Tag', 'figPanZoom');
	RT = findobj_hidden(hToolbar_Children, 'Tag', 'figROITool');
	MV = findobj_hidden(hToolbar_Children, 'Tag', 'figViewImages');
	PM = findobj_hidden(hToolbar_Children, 'Tag', 'figPointTool');
	RotT = findobj_hidden(hToolbar_Children, 'Tag', 'figRotateTool');
	Prof = findobj_hidden(hToolbar_Children, 'Tag', 'figProfileTool');

	old_ToolHandles  =     [Rot3D, ZoomO, ZoomI,PZ,RT,MV,PM,RotT,Prof];
	old_ToolEnables  = get([Rot3D, ZoomO, ZoomI,PZ,RT,MV,PM,RotT,Prof], 'Enable');
	old_ToolStates   = get([Rot3D, ZoomO, ZoomI,PZ,RT,MV,PM,RotT,Prof], 'State');
	
	for i = 1:length(old_ToolHandles)
		if strcmp(old_ToolStates(i) , 'on')			
			set(old_ToolHandles(i), 'State', 'Off');
		end;
		set(old_ToolHandles(i), 'Enable', 'Off');
	end;
        %LFG
        %enable save_prefs tool button
        SP = findobj(hToolbar_Children, 'Tag', 'figSavePrefsTool');
        set(SP,'Enable','On');
end;

% Store initial state of all axes in current figure for reset
h_all_axes = flipud(findobj(fig,'Type','Axes'));
for i = 1:length(h_all_axes)
	all_clims(i,:) = get(h_all_axes(i),'Clim');
end;

% obtain current axis
set(0, 'CurrentFigure', fig);
h_axes= gca;
if isempty(h_axes), h_axes = h_all_axes(1); end;

% Record and store previous WBDF etc to restore state after WL is done. 
%previous_BDF = get(fig, 'WindowButtonDownFcn');
old_WBDF     = get(fig, 'WindowButtonDownFcn');
old_WBMF     = get(fig, 'WindowButtonMotionFcn');
old_WBUF     = get(fig, 'WindowButtonUpFcn');
old_UserData = get(fig, 'UserData');
old_CRF      = get(fig, 'Closerequestfcn');

% Is this needed?
%set(0, 'ShowHiddenHandles', 'On');
	
% Start WL GUI
fig2_old = findobj_hidden('Tag', 'WL_figure');
% close the old WL figure to avoid conflicts
if ~isempty(fig2_old) close(fig2_old);end;
pause(0.5);

% open new figure
%LFG
fig2_file = 'WL_tool_figure.fig';
fig2 = openfig(fig2_file,'reuse');
optional_uicontrols = {'Apply_to_popupmenu', 'Value'};
set(SP,'Userdata',{fig2, fig2_file, optional_uicontrols});

set(fig, 'WindowButtonDownFcn',   'WL_tool(''Adjust_On'');');
set(fig, 'WindowButtonUpFcn',     'WL_tool(''Adjust_Window_Level_For_All''); ');
set(fig, 'WindowButtonMotionFcn', '');

% Draw faster and without flashes
set(fig, 'Closerequestfcn', [ old_CRF , ',WL_tool(''Close_Parent_Figure'')']);
set(fig, 'Renderer', 'zbuffer');
set(0, 'CurrentFigure', fig);
set(gca, 'Drawmode', 'Fast');



% Generate a structure of handles to pass to callbacks, and store it. 
handlesWL = guihandles(fig2);
handlesWL.Parent_Figure = fig;
handlesWL.All_Axes = h_all_axes;
guidata(fig2,handlesWL);

% determine current figure's colormap
s = get(handlesWL.Colormap_popupmenu,'String');
cmap_value = Find_Colormap(get(fig, 'Colormap'), {s{1:end-1}});
set(handlesWL.Colormap_popupmenu, 'UserData' , {fig , get(fig,'Colormap') , cmap_value} ); %%%put current value here
set(handlesWL.Colormap_popupmenu, 'Value' , length(s));
Set_Colormap(handlesWL.Colormap_popupmenu);

close_str = [ ...
		' set(0,''ShowHiddenHandles'',''On'');' ...
		' hNewButton = findobj(''Tag'', ''figWindowLevel'');' ...
		' if strcmp(get(hNewButton, ''Type''), ''uitoggletool''),'....
		' set(hNewButton, ''State'', ''off'' );' ...
		' else,  ' ...
		' WL_tool(''Deactivate_Window_Level'',hNewButton);'...
		' set(hNewButton, ''Value'', 0);' ...
		' end;'...
		' set(0,''ShowHiddenHandles'',''Off'');'];
set(fig2, 'Name', 'WL Tool',...
	'closerequestfcn', close_str);    

% store the figure's old infor within the fig's own userdata
set(fig, 'UserData', {fig2, old_WBDF, old_WBMF, old_WBUF, old_UserData,old_CRF, ...
		old_ToolEnables, old_ToolHandles, old_ToolStates});

% setup sliders and edit boxes
Clim = get(h_axes,'clim');
window =  Clim(2) - Clim(1)    ; 
level =  (Clim(2) + Clim(1) )/2;

set(handlesWL.Reset_pushbutton, 'UserData', {h_all_axes, all_clims, h_axes }, 'Enable', 'Off');
set(handlesWL.Window_value_edit,'Enable', 'Off');
set(handlesWL.Level_value_edit,'Enable', 'Off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Deactivate_Window_Level(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('WL_tool: Deactivate_Window_Level');

if nargin == 0,
	old_SHH = get(0, 'ShowHiddenHandles');
	set(0, 'ShowHiddenHandles', 'On');    
	hNewButton = gcbo;
	set(findobj('Tag', 'menuWindowLevel'),'checked', 'Off');
	set(0, 'ShowHiddenHandles', old_SHH);
else
	hNewButton = varargin{1};
end;

% Reactivate other buttons
fig = get(hNewButton, 'Parent');
if ~strcmp(get(fig, 'Type'), 'figure'),
	fig = get(fig, 'Parent');
end

hToolbar = findall(fig, 'type', 'uitoolbar');
% if ~isempty(hToolbar)
% 	hToolbar_Children = get(hToolbar, 'Children');
% 	set(findobj(hToolbar_Children,'Tag', 'figToolRotate3D'),'Enable', 'On');
% 	set(findobj(hToolbar_Children,'Tag', 'figToolZoomOut'),'Enable', 'On');
% 	set(findobj(hToolbar_Children,'Tag', 'figToolZoomIn'),'Enable', 'On');  
% end;

% Restore old BDFs
old_info= get(fig,'UserData');
set(fig, 'WindowButtonDownFcn', old_info{2});
set(fig, 'WindowButtonUpFcn', old_info{3});
set(fig, 'WindowButtonMotionFcn', old_info{4});
set(fig, 'UserData', old_info{5});
set(fig, 'CloseRequestFcn', old_info{6});
old_ToolEnables = old_info{7};
old_ToolHandles = old_info{8};
old_ToolStates  = old_info{9};

fig2 = old_info{1};
try
	set(fig2, 'CloseRequestFcn', 'closereq');
	close(fig2); 
catch
	delete(fig2);
end;    

for i = 1:length(old_ToolHandles)
	try
		set(old_ToolHandles(i), 'Enable', old_ToolEnables{i}, 'State', old_ToolStates{i});
	end;
end;
%LFG
%disable save_prefs tool button
SP = findobj(get(hToolbar, 'Children'), 'Tag', 'figSavePrefsTool');
set(SP,'Enable','Off');

%set(0, 'ShowHiddenHandles', 'Off');


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Adjust_On;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execute once at the beggining of a drag cycle
set(gcf, 'WindowButtonMotionFcn', 'WL_tool(''Adjust_Window_Level'');');
point = get(gca,'CurrentPoint');
Clim = get(gca, 'Clim');
set(findobj_hidden(gcf, 'Tag', 'figWindowLevel'), 'UserData',[point(1,1) point(1,2), Clim]);   
Adjust_Window_Level;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Adjust_Window_Level;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
h_axes = gca;
point = get(h_axes,'CurrentPoint');
fig = get(h_axes,'Parent');
ref_coor = get( findobj_hidden(fig,'Tag', 'figWindowLevel'),'UserData');

clim= ref_coor(3:4);
xlim= get(gca,'Xlim');
ylim= get(gca,'Ylim');

window = (clim(2) - clim(1) )  ;
level =  (clim(2) + clim(1) )/2;

% Use fraction  i.e. relative to position to the originally clicked point
% to determine the change in window and level
deltas = point(1,1:2) - ref_coor(1:2);

% To change WL sensitivity to position, change exponento to bigger/ smaller odd number 
sensitivity_factor = 3; 
new_level =   level  + level  * (deltas(2) / diff(ylim))^sensitivity_factor;
new_window =  window + window * (deltas(1) / diff(xlim))^sensitivity_factor;

% make sure clims stay ascending
if (new_window < 0) new_window = 0.1; end;
set(h_axes, 'Clim', [new_level - new_window/2 , new_level + new_window/2]);
set(findobj_hidden('Tag', 'Apply_to_popupmenu'),'UserData', { [new_level, new_window], h_axes, fig});

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Adjust_Window_Level_For_All;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execute once after window/level is done
% Check to see if all images in slice should be rescaled
fig2 = findobj_hidden('Tag', 'WL_figure');

handlesWL = guidata(fig2);

apply_all = get(handlesWL.Apply_to_popupmenu,'Value');
new_level = get(handlesWL.Apply_to_popupmenu,'UserData');
h_axes= gca;
h_axes_index = find(handlesWL.All_Axes==h_axes);

new_level = new_level{1};
new_window = new_level(2);
new_level = new_level(1);

if apply_all == 1
	% do _nothing
elseif apply_all == 2
	% All
	set(handlesWL.All_Axes, 'Clim', [new_level - new_window/2 , new_level + new_window/2]);
elseif apply_all == 3
	% odd
	if (mod(h_axes_index,2))
		set(handlesWL.All_Axes(1:2:end), 'Clim', [new_level - new_window/2 , new_level + new_window/2]);
	else
		set(handlesWL.All_Axes(2:2:end), 'Clim', [new_level - new_window/2 , new_level + new_window/2]);
	end;
elseif apply_all == 4
	% 1:current
	set(handlesWL.All_Axes(1:h_axes_index), 'Clim', [new_level - new_window/2 , new_level + new_window/2]);
elseif apply_all == 5
	% current:end
	set(handlesWL.All_Axes(h_axes_index:end), 'Clim', [new_level - new_window/2 , new_level + new_window/2]);
end;


userdata = get(handlesWL.Reset_pushbutton, 'UserData');
userdata{3} = h_axes;
set(handlesWL.Reset_pushbutton, 'UserData', userdata, 'Enable', 'On');
set(handlesWL.Window_value_edit, 'Enable', 'On');
set(handlesWL.Level_value_edit, 'Enable', 'On');

% now update editable text boxes
Update_Window_Level(handlesWL, new_window, new_level);
figure(fig2); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Update_Window_Level(handlesWL, window, level );
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
set(handlesWL.Window_value_edit, 'String', num2str(window,5));
set(handlesWL.Level_value_edit,  'String', num2str(level,5) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Set_Colormap(Colormap_popupmenu);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change the colormap to the one specified by the popupmenu
% Use the same size of the original colormap as stored in
% the popupmenu's userdata = { fig , cmap , old_value}
%disp('WL_tool: Set_Colormap');
new_value = get(Colormap_popupmenu, 'Value');
cmap  = get(Colormap_popupmenu,'String');
cmap2 = cmap{new_value};
userdata= get(Colormap_popupmenu,'UserData');
old_value = userdata{3};

% now use the previous pmenu value to put the old string back on top...
% get(pmenu,'Value')
if strcmp(cmap2, 'More...')
	s = {'Jet', 'Gray','Hsv','Hot','Bone','Copper','Pink','White','Flag','Lines',...
			'Colorcube','Vga','Prism','Cool','Autumn','Spring','Winter','Summer','Less...'};
	set(Colormap_popupmenu,'String', s);
	val = strmatch(cmap{old_value},s);
	
	if old_value>2
		set(Colormap_popupmenu,'Value', val ) ;
	else
		set(Colormap_popupmenu,'Value', old_value);
	end;
	
elseif strcmp(cmap2,'Less...')
	% if current cmap is not jet or gray, keep name on string
	if old_value >2
		set(Colormap_popupmenu,'String', {'Jet', 'Gray', cmap{old_value}, 'More...'});
		set(Colormap_popupmenu,'Value', 3); 
		userdata{3} = 3;
	else
		set(Colormap_popupmenu, 'String', {'Jet', 'Gray','More...'});
		set(Colormap_popupmenu, 'Value', old_value);
		userdata{3} = old_value;
	end;    
	
else
	if strcmp(cmap2,'Vga') % vga does takes only 16 colors...
		eval(['set(', num2str(userdata{1}) , ', ''colormap'' , colormap(' ,cmap2,'))' ]);        
	else
		eval(['set(', num2str(userdata{1}, '%1.15g') , ', ''colormap'' , colormap(' ,cmap2, '(' num2str(size(userdata{2},1)), ')))' ]);
	end;
	userdata{3} = new_value;
	
end;
set(Colormap_popupmenu,'UserData', userdata);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function WL_Reset;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2 = findobj_hidden('Tag', 'WL_figure');
handlesWL = guidata(fig2);
old_axes_info = get(handlesWL.Reset_pushbutton, 'UserData');
apply_all = get(handlesWL.Apply_to_popupmenu,'Value');
h_axes = handlesWL.All_Axes;

clims = old_axes_info{2};
h_current_axes = old_axes_info{3};
current_axes_index = find(h_axes==h_current_axes);

if apply_all == 1
	h_axes_of_interest = h_current_axes;
	indexes = current_axes_index;
elseif apply_all == 2
	h_axes_of_interest = handlesWL.All_Axes;
	indexes = 1:length(h_axes_of_interest);
elseif apply_all == 3
	if (mod(current_axes_index,2)), 
		h_axes_of_interest = handlesWL.All_Axes(1:2:end);
		indexes = 1:2:length(handlesWL.All_Axes);
	else,
		h_axes_of_interest = handlesWL.All_Axes(2:2:end);
		indexes = 2:2:length(handlesWL.All_Axes);		
	end;
elseif apply_all == 4
	h_axes_of_interest = handlesWL.All_Axes(1:current_axes_index);
	indexes = 1:length(h_axes_of_interest);
elseif apply_all == 5
	h_axes_of_interest = handlesWL.All_Axes(current_axes_index:end);
	indexes = find(h_axes_of_interest(1)==handlesWL.All_Axes): length(handlesWL.All_Axes);
end

for i = 1:length(h_axes_of_interest)
	set(h_axes_of_interest(i),'Clim', clims(indexes(i),:));
end;

% now update sliders
window = (clims(current_axes_index,2)-clims(current_axes_index,1));
level =  (clims(current_axes_index,2)+clims(current_axes_index,1))/2;
Update_Window_Level(handlesWL, window, level);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Auto_WL_Reset;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2 = findobj_hidden('Tag', 'WL_figure');
handlesWL = guidata(fig2);
old_axes_info = get(handlesWL.Reset_pushbutton, 'UserData');
apply_all = get(handlesWL.Apply_to_popupmenu,'Value');

h_axes = handlesWL.All_Axes;
clims = old_axes_info{2};
h_current_axes = old_axes_info{3};
current_axes_index = find(h_axes==h_current_axes);

if apply_all == 1
	h_axes_of_interest = h_current_axes;
elseif apply_all == 2
	h_axes_of_interest = handlesWL.All_Axes;
elseif apply_all == 3
	if (mod(current_axes_index,2)), 
		h_axes_of_interest = handlesWL.All_Axes(1:2:end);
	else,
		h_axes_of_interest = handlesWL.All_Axes(2:2:end);
	end;
elseif apply_all == 4
	h_axes_of_interest = handlesWL.All_Axes(1:current_axes_index);
elseif apply_all == 5
	h_axes_of_interest = handlesWL.All_Axes(current_axes_index:end);
end

for i = 1:length(h_axes_of_interest)
	%set(h_axes(i),'Climmode', 'auto');
	xlim = get(h_axes_of_interest(i),'Xlim'); ylim = get(h_axes_of_interest(i),'Ylim');
	x1 = ceil(xlim(1)); x2 = floor(xlim(2));
	y1 = ceil(ylim(1)); y2 = floor(ylim(2));
	c = get(findobj(h_axes_of_interest(i),'type','image'),'CData');
	cmin = min(min(c(y1:y2,x1:x2)));
	cmax = max(max(c(y1:y2,x1:x2)));
	set(h_axes_of_interest(i),'Clim',[cmin cmax]);
end;

% now update sliders
window = (clims(current_axes_index,2)-clims(current_axes_index,1));
level =  (clims(current_axes_index,2)+clims(current_axes_index,1))/2;
Update_Window_Level(handlesWL, window, level);



%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Edit_Adjust;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig2 = findobj_hidden('Tag', 'WL_figure');
handlesWL = guidata(fig2);

apply_all = get(handlesWL.Apply_to_popupmenu,'Value');
old_axes_info = get(handlesWL.Reset_pushbutton, 'UserData');
h_axes = handlesWL.All_Axes;
h_current_axes = old_axes_info{3};
current_axes_index = find(h_axes==h_current_axes);

window = str2num(get(handlesWL.Window_value_edit,'String'));
level =  str2num(get(handlesWL.Level_value_edit,'String'));

% update the current graph or all the graphs...

if apply_all == 1
	h_axes_of_interest = h_current_axes;
elseif apply_all == 2
	h_axes_of_interest = handlesWL.All_Axes;
elseif apply_all == 3
	if (mod(current_axes_index,2)), 
		h_axes_of_interest = handlesWL.All_Axes(1:2:end);
	else,
		h_axes_of_interest = handlesWL.All_Axes(2:2:end);
	end;
elseif apply_all == 4
	h_axes_of_interest = handlesWL.All_Axes(1:current_axes_index);
elseif apply_all == 5
	h_axes_of_interest = handlesWL.All_Axes(current_axes_index:end);
end

for i = 1:length(h_axes_of_interest)
	set(h_axes_of_interest(i),'Clim', [level-window/2, level+window/2]);
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Menu_Window_Level;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

hNewMenu = gcbo;
checked=  umtoggle(hNewMenu);
hNewButton = get(hNewMenu, 'userdata');

if ~checked
	% turn off button
	%Deactivate_Window_Level(hNewButton);
	set(hNewMenu, 'Checked', 'off');
	set(hNewButton, 'State', 'off' );
else
	%Activate_Window_Level(hNewButton);
	set(hNewMenu, 'Checked', 'on');
	set(hNewButton, 'State', 'on' );
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Close_Parent_Figure;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to make sure that if parent figure is closed, 
% the ROI info and ROI Tool are closed too.
set(findobj_hidden('Tag', 'WL_figure'), 'Closerequestfcn', 'closereq');
try 
	close(findobj_hidden('Tag','WL_figure'));
catch
	delete(findobj_hidden('Tag', 'WL_figure'));
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Cmap_Value = Find_Colormap(Current_Cmap, s);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine the current figure's colomrap by comparing it with 
% the established colormaps of the same size
%disp('WL_tool: Find_Colormap');
Cmap_Value = 1; % default
for i = 1:length(s)
	if ~strcmp(lower(s{i}),'vga')
		eval(['test_cmap = ',  lower(s{i}), '(',num2str(size(Current_Cmap,1)),');']);
		%diffy = Current_Cmap - test_cmap;
		%c_dffy = cumsum(diffy)
		if isempty(find(test_cmap - Current_Cmap))
			Cmap_Value = i;
			return;	
		end;
	elseif (size(Current_Cmap,1)==16) % vga is only 16 colors
		eval(['test_cmap = ',  lower(s{i}), ';']);
		if isempty(find(test_cmap - Current_Cmap))
			Cmap_Value = i;
			return;	
		end;
	end;	
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function h= findobj_hidden(Handle, Property, Value);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('WL_tool: findobj_hidden');
old_SHH = get(0, 'ShowHiddenHandles');
set(0, 'ShowHiddenHandles', 'On');
if nargin <3
	h = findobj(Handle, Property);
else
	h = findobj(Handle, Property, Value);
end;
set(0, 'ShowHiddenHandles', old_SHH);


