function PZ_tool(varargin);
%function PZ_tool(varargin);
% Pan - Zoom Tool to be used with imagescn. 
% Usage: Pz_tool;
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
end

%global DB; DB = 1;

switch Action
case 'New'
    Create_New_Button;

case 'Activate_Pan_Zoom'
    Activate_Pan_Zoom;
    
case 'Deactivate_Pan_Zoom'
    Deactivate_Pan_Zoom(varargin{2:end});
        
case 'Adjust_On'
   %Entry
   Turn_Adjust_Pan_On;
case 'Adjust_Pan'
   % Cycle
    Adjust_Pan;
case 'Adjust_Pan_For_All'
    % Exit
    set(gcf, 'WindowButtonMotionFcn', ' ');
    Adjust_Pan_For_All;    

case 'Switch_Pan_Zoom'
    % Change from active panning to active zooming and vice versa
    Switch_Pan_Zoom;
   
case 'Zoom'
    Apply_Zoom_Factor;
    
case 'Done_Zoom'
    Done_Zoom;

case 'PZ_Reset'
    PZ_Reset;

case 'Auto_PZ_Reset'
    Auto_PZ_Reset;
    
    
    
case 'Menu_Pan_Zoom'
    Menu_Pan_Zoom;
    
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
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

fig = gcf;

% Find handle for current image toolbar and menubar
hToolbar = findall(fig, 'type', 'uitoolbar', 'Tag','FigureToolBar' );
hToolMenu = findall(fig, 'Label', '&Tools');

if ~isempty(hToolbar) & isempty(findobj(hToolbar, 'Tag', 'figPanZoom'))
	hToolbar_Children = get(hToolbar, 'Children');

	% The default button size is 15 x 16 x 3. Create Button Image
   button_size_x= 16;
   button_image = NaN* zeros(15,button_size_x);
    
   f= [...
     8     9    22    25    33    34    37    41    42,...
    47    50    51    53    58    62    67    68    69,...
    74    78    79    90    92    93    94    95    96,...
    97   106   121   137   138   139   140   141   142,...
   152   167   183   184   185   186   187   188   194,...
   195   200   206   207   208   214   219   220   221,...
   230   231   232   233];
   
   button_image(f) = 0;
   button_image = repmat(button_image, [1,1,3]);

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
      'OnCallback', 'PZ_tool(''Activate_Pan_Zoom'')',...
      'OffCallback', 'PZ_tool(''Deactivate_Pan_Zoom'')',...
      'Tag', 'figPanZoom', ...
      'TooltipString', 'Pan and Zoom figure',...
      'UserData', [], ...
      'Enable', 'on');   
end;

% If the menubar exists, create menu item
if ~isempty(hToolMenu) & isempty(findobj(hToolMenu, 'Tag', 'menuPanZoom'))
  hWindowLevelMenu = findobj(hToolMenu, 'Tag', 'menuWindowLevel');
  hPanZoomMenu     = findobj(hToolMenu, 'Tag', 'menuPanZoom');
  hROIToolMenu     = findobj(hToolMenu, 'Tag', 'menuROITool');
  hViewImageMenu   = findobj(hToolMenu, 'Tag', 'menuViewImages');
  hPointToolMenu   = findobj(hToolMenu, 'Tag', 'menuPointTool');
  hRotateToolMenu  = findobj(hToolMenu, 'Tag', 'menuRotateTool');
  hProfileToolMenu = findobj(hToolMenu, 'Tag', 'menuProfileTool');
	
  position = 9;
  separator = 'On';
  hMenus = [ hWindowLevelMenu, hROIToolMenu, hViewImageMenu, hPointToolMenu, hRotateToolMenu,hProfileToolMenu ];

  if length(hMenus>0) 
	  position = position + length(hMenus);
	  separator = 'Off';
  end;
  
  hNewMenu = uimenu(hToolMenu,'Position', position);
  set(hNewMenu, 'Tag', 'menuPanZoom','Label',...
      'Pan and Zoom',...
      'CallBack', 'PZ_tool(''Menu_Pan_Zoom'')',...
      'Separator', separator,...
      'UserData', hNewButton...
  ); 
  
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Activate_Pan_Zoom(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

if nargin ==0
    set(0, 'ShowHiddenHandles', 'On');
    hNewButton = gcbo;
    set(findobj('Tag', 'menuPanZoom'),'checked', 'on');
else
    hNewButton = varargin{1};
end;

% allows for calls from buttons other than those in toolbar
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
	Rot3D = findobj(hToolbar_Children,'Tag', 'figToolRotate3D');
	ZoomO = findobj(hToolbar_Children,'Tag', 'figToolZoomOut');
	ZoomI = findobj(hToolbar_Children,'Tag', 'figToolZoomIn');

	% try to disable other tools buttons - if they exist
	WL = findobj(hToolbar_Children, 'Tag', 'figWindowLevel');
	PZ = findobj(hToolbar_Children,'Tag', 'figPanZoom');
	RT = findobj(hToolbar_Children,'Tag', 'figROITool');
	MV = findobj(hToolbar_Children,'Tag', 'figViewImages');
	PM = findobj(hToolbar_Children,'Tag', 'figPointTool');
	RotT = findobj(hToolbar_Children,'Tag', 'figRotateTool');
	Prof = findobj(hToolbar_Children, 'Tag', 'figProfileTool');	
	
	old_ToolHandles  =     [Rot3D, ZoomO, ZoomI,WL,RT,MV,PM, RotT,Prof];
	old_ToolEnables  = get([Rot3D, ZoomO, ZoomI,WL,RT,MV,PM, RotT,Prof], 'Enable');
	old_ToolStates   = get([Rot3D, ZoomO, ZoomI,WL,RT,MV,PM, RotT,Prof], 'State');
	
	for i = 1:length(old_ToolHandles)
		if strcmp(old_ToolStates(i) , 'on')			
			set(old_ToolHandles(i), 'State', 'Off');
		end;
		set(old_ToolHandles(i), 'Enable', 'Off');
	end;
end;


% Start PZ GUI
fig2_old = findobj('Tag', 'PZ_figure');
% close the old WL figure to avoid conflicts
if ~isempty(fig2_old) close(fig2_old);end;

% open new figure
fig2 = openfig('PZ_tool_figure','reuse');

% Generate a structure of handles to pass to callbacks, and store it. 
handlesPZ = guihandles(fig2);
guidata(fig2,handlesPZ);

close_str = [ 'hNewButton = findobj(''Tag'', ''figPanZoom'');' ...
        ' if strcmp(get(hNewButton, ''Type''), ''uitoggletool''),'....
        ' set(hNewButton, ''State'', ''off'' );' ...
        ' else,  ' ...
        ' PZ_tool(''Deactivate_Pan_Zoom'',hNewButton);'...
        ' set(hNewButton, ''Value'', 0);',...
        ' end;' ];

set(fig2, 'Name', 'PZ Tool',...
    'closerequestfcn', close_str);


% change the pointer and store the old pointer data
% define the black and white parts for both the pointer shapes
%
f_black_open = [...
     8     9    22    25    33    34    37    41    42    47    50    51    53,...
    58    62    67    68    74    78    79    90    92    93    94    95    96,...
    97   106   121   137   138   139   140   141   142   152   167   183   184,...
   185   186   187   188   194   195   200   206   207   208   214   219   220,...
   221   230   231   232   233];

f_white_open =[...
    23    24    38    39    40    48    49    54    55    56    57    63    64,...
    65    66    69    70    71    72    73    80    81    82    83    84    85,...
    86    87    88    89    98    99   100   101   102   103   104   105   107,...
   108   109   110   111   112   113   114   115   116   117   118   119   120,...
   122   123   124   125   126   127   128   129   130   131   132   133   134,...
   135   143   144   145   146   147   148   149   150   153   154   155   156,...
   157   158   159   160   161   162   163   164   165   168   169   170   171,...
   172   173   174   175   176   177   178   179   180   189   190   191   192,...
   193   201   202   203   204   205   215   216   217   218];

f_black_closed = [...
    41    42    52    53    55    58    66    69    70    74    81    90    97,...
   111   126   142   156   171   187   188   194   195   202   206   207   208,...
   218   219   220   221];

f_white_closed = [...
    56    57    67    68    71    72    73    82    83    84    85    86    87,...
    88    89    98    99   100   101   102   103   104   105   112   113   114,...
   115   116   117   118   119   120   127   128   129   130   131   132   133,...
   134   135   143   144   145   146   147   148   149   150   157   158   159,...
   160   161   162   163   164   165   172   173   174   175   176   177   178,...
   179   180   189   190   191   192   193   203   204   205];


pointer_image = NaN*zeros(15,16);
pointer_image(f_black_open) = 1;
pointer_image(f_white_open) = 2;
% repeat the last line to make the pointer data 16x16
pointer_image = cat(1,pointer_image,pointer_image(15,:)); 

closed_pointer_image = NaN*zeros(15,16);
closed_pointer_image(f_black_closed) = 1;
closed_pointer_image(f_white_closed) = 2;
closed_pointer_image = cat(1,closed_pointer_image,closed_pointer_image(15,:)); 

old_pointer      = get(fig, 'Pointer');
old_pointer_data = get(fig, 'PointerShapeCData');
set(fig,'Pointer', 'Custom', 'PointerShapeCData', pointer_image);,
set(hNewButton, 'UserData', {[], closed_pointer_image, pointer_image});

% Record and store previous WBDF etc to restore state after PZ is done. 
old_WBDF = get(fig, 'WindowButtonDownFcn');
old_WBMF = get(fig, 'WindowButtonMotionFcn');
old_WBUF = get(fig, 'WindowButtonUpFcn');
old_UserData = get(fig, 'UserData');
old_CRF = get(fig, 'Closerequestfcn');

% Store initial state of all axes in current figure for reset
h_all_axes = findobj(fig,'Type','Axes');
for i = 1:length(h_all_axes)
    all_xlims(i,:) = get(h_all_axes(i),'Xlim');
    all_ylims(i,:) = get(h_all_axes(i),'Ylim');
end;

h_axes = h_all_axes(end);
im = get(findobj(h_axes, 'Type', 'Image'),'Cdata');
zoom_factor = max([size(im,2)/diff(all_xlims(end,:)),  size(im,1)/diff(all_ylims(end,:))]);

set(fig, 'CurrentAxes', h_axes);

% Draw faster and without flashes
set(fig, 'Closerequestfcn', [ old_CRF , ',PZ_tool(''Close_Parent_Figure'')']);
set(fig, 'Renderer', 'zbuffer');
set(0, 'ShowHiddenHandles', 'On', 'CurrentFigure', fig);
set(gca,'Drawmode', 'Fast');

% store the figure's old infor within the fig's own userdata
set(fig, 'UserData', {fig2, old_WBDF, old_WBMF, old_WBUF, old_UserData,...
        old_pointer, old_pointer_data, old_CRF, ...
		old_ToolEnables,old_ToolHandles, old_ToolStates});

%set(fig, 'WindowButtonDownFcn', '');  %entry
%set(fig, 'WindowButtonUpFcn','');  %exit
set(fig, 'WindowButtonDownFcn', 'PZ_tool(''Adjust_On'');');  %entry
set(fig, 'WindowButtonUpFcn','PZ_tool(''Adjust_Pan_For_All''); ');  %exit
set(fig, 'WindowButtonMotionFcn', '');  % entry function sets the WBMF

% store all relevant info for faster use during calls
set(handlesPZ.Reset_pushbutton, 'UserData', {fig, fig2, h_all_axes, all_xlims, all_ylims, h_axes }, 'Enable', 'Off');
set(handlesPZ.Zoom_value_edit, 'String', num2str(zoom_factor,3));
%set(handlesPZ.Pan_radiobutton, 'Value', 0);
%set(handlesPZ.Zoom_radiobutton, 'Value', 1);

%im = get(findobj(h_axes, 'Type', 'image'), 'Cdata');
Switch_Pan_Zoom;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Deactivate_Pan_Zoom(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

if nargin ==0
    set(0, 'ShowHiddenHandles', 'On');    
    hNewButton = gcbo;
    set(findobj('Tag', 'menuPanZoom'),'checked', 'Off');
else
    hNewButton = varargin{1};
end;
    
% Reactivate other buttons
fig = get(hNewButton, 'Parent');
if ~strcmp(get(fig, 'Type'), 'figure'),
    fig = get(fig, 'Parent');
end

hToolbar = findall(fig, 'type', 'uitoolbar');
if ~isempty(hToolbar)
    hToolbar_Children = get(hToolbar, 'Children');
    set(findobj(hToolbar_Children,'Tag', 'figToolRotate3D'),'Enable', 'On');
    set(findobj(hToolbar_Children,'Tag', 'figToolZoomOut'),'Enable', 'On');
    set(findobj(hToolbar_Children,'Tag', 'figToolZoomIn'),'Enable', 'On');

end;

myzoom('off');

% Restore old BDFs
old_info= get(fig,'UserData');
set(fig, 'WindowButtonDownFcn', old_info{2});
set(fig, 'WindowButtonUpFcn', old_info{3});
set(fig, 'WindowButtonMotionFcn', old_info{4});


% Restore old Pointer and UserData
set(fig, 'UserData', old_info{5});
set(fig, 'Pointer' , old_info{6});
set(fig, 'PointerShapeCData', old_info{7});
set(fig, 'CloseRequestFcn', old_info{8});
old_ToolEnables  = old_info{9};
old_ToolHandles = old_info{10};
old_ToolStates = old_info{11};

fig2 = old_info{1};
try
	set(fig2, 'CloseRequestFcn', 'closereq');
	close(fig2); 
catch
	delete(fig2);
end;    

for i = 1:length(old_ToolHandles)
	try
		set(old_ToolHandles(i), 'Enable', old_ToolEnables{i}, 'State', old_ToolStates{i});	catch
	end;
end;

set(0, 'ShowHiddenHandles', 'Off');


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Turn_Adjust_Pan_On;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

% Execute once at the beggining of a drag cycle
fig = gcf;
set(fig, 'WindowButtonMotionFcn', 'PZ_tool(''Adjust_Pan'');');

% change the pointer to closed hand
hNewButton = findobj(gcf, 'Tag', 'figPanZoom');
data = get(hNewButton,'Userdata');
set(fig,'PointerShapeCData', data{2});
point = get(gca,'CurrentPoint');
data{1} = [point(1,1) point(1,2)];
% now store current point as reference
set(hNewButton, 'UserData',data);  
Adjust_Pan;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Adjust_Pan;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

h_axes = gca;
point = get(h_axes,'CurrentPoint');
ref_coor = get( findobj(get(h_axes,'Parent'),'Tag', 'figPanZoom'),'UserData');

xlim= get(gca,'Xlim');
ylim= get(gca,'Ylim');

% Use fraction  i.e. relative to position to the originally clicked point
% to determine the change in window and level
deltas = point(1,1:2) - ref_coor{1};

xlim =xlim - deltas(1);
ylim =ylim - deltas(2);

% set the xlims and the ylims after motion
fig = get(h_axes, 'Parent');
set(h_axes, 'Xlim', xlim, 'Ylim', ylim);
set(findobj('Tag', 'Apply_radiobutton'),'UserData', { xlim, ylim, h_axes});


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Adjust_Pan_For_All;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Execute once after panning is done
% Check to see if all images in slice should be rescaled
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

fig2 = findobj('Tag', 'PZ_figure');
handlesPZ = guidata(fig2);
apply_all = get(handlesPZ.Apply_radiobutton,'Value');
most_current_data = get(handlesPZ.Apply_radiobutton, 'UserData');
userdata = get(handlesPZ.Reset_pushbutton, 'UserData');
h_axes = gca;

fig = userdata{1};

if apply_all
    all_axes = userdata{3};
    for i =1:length(all_axes)
        set(fig, 'CurrentAxes', all_axes(i));
        set(0,'CurrentFigure', fig);
        set(all_axes(i), 'xlim', most_current_data{1}, 'ylim', most_current_data{2});
    end;
        h_axes = most_current_data{3};
else
    set(0,'CurrentFigure', fig);
    set(h_axes, 'xlim', most_current_data{1}, 'ylim', most_current_data{2});        
end;
userdata{6} = h_axes;
set(handlesPZ.Reset_pushbutton, 'UserData', userdata, 'Enable', 'On');
data= get(findobj(fig, 'Tag', 'figPanZoom'),'UserData');
set(fig, 'PointerShapeCData', data{3});
figure(fig2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Apply_Zoom_Factor;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

fig2 = findobj('Tag', 'PZ_figure');
handlesPZ = guidata(fig2);
apply_all = get(handlesPZ.Apply_radiobutton,'Value');
zoom_factor = str2num(get(handlesPZ.Zoom_value_edit,'String'));
userdata = get(handlesPZ.Reset_pushbutton, 'UserData');
fig = userdata{1};

all_axes = userdata{3};
if ~apply_all, all_axes = userdata{6}; end;

im = get(findobj(all_axes(1), 'type', 'image'), 'cdata');
xsize = size(im,2);
ysize = size(im,1);

for i =1:length(all_axes)
	ax = get(all_axes(i), {'xlim', 'ylim'});
	cx = mean(ax{1}); cy = mean(ax{2});
	lx = (xsize)/zoom_factor/2;
	ly = (ysize)/zoom_factor/2;
	
	set(all_axes(i), 'xlim', [-lx, lx]+cx, 'ylim', [-ly, ly]+cy);		
	%myzoom(zoom_factor);	%set(fig, 'CurrentAxes', all_axes(i));
	%set(0,'CurrentFigure', fig);
end;


% save last axis with an action
set(handlesPZ.Reset_pushbutton, 'UserData', userdata, 'Enable', 'On');


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function PZ_Reset;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to reset the axes limits to whatever they were upon
% startup of the tool
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;


fig2 = findobj('Tag', 'PZ_figure');
handlesPZ = guidata(fig2);
old_axes_info = get(handlesPZ.Reset_pushbutton, 'UserData');
reset_all_axes = get(handlesPZ.Apply_radiobutton, 'Value');
fig = old_axes_info{1};
h_all_axes = old_axes_info{3};
xlims = old_axes_info{4};
ylims = old_axes_info{5};
last_axes = old_axes_info{6};

current_axes_index = find(h_all_axes==last_axes);
im = get(findobj(h_all_axes(current_axes_index), 'Type', 'Image'), 'Cdata');

% recalculate the original zoom factors
zoom_factor_x = size(im,2) / diff(xlims(current_axes_index,:));
zoom_factor_y = size(im,1) / diff(ylims(current_axes_index,:));

if reset_all_axes
    for i = 1:length(h_all_axes)
        set(fig, 'CurrentAxes', h_all_axes(i));
        set(0,'CurrentFigure', fig);
        set(h_all_axes(i),'Xlim', xlims(i,:),'ylim',ylims(i,:))
    end;
else
    %    set(h_all_axes(current_axes_index),'xlims', xlims(i,:), 'ylims',ylims(i,:));
    set(fig, 'CurrentAxes', last_axes);
    set(0,'CurrentFigure', fig);
    set(last_axes, 'Xlim', xlims(current_axes_index,:), 'YLim', ylims(current_axes_index,:));
end;
set(handlesPZ.Zoom_value_edit,'String', num2str(1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Auto_PZ_Reset;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to automatically set the limits - ie use the complete
% cdata 
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

fig2 = findobj('Tag', 'PZ_figure');
handlesPZ = guidata(fig2);
old_axes_info = get(handlesPZ.Reset_pushbutton, 'UserData');
reset_all_axes = get(handlesPZ.Apply_radiobutton, 'Value');
fig = old_axes_info{1};
h_all_axes = old_axes_info{3};
xlims = old_axes_info{4};
ylims = old_axes_info{5};
last_axes = old_axes_info{6};
current_axes_index = find(h_all_axes==last_axes);

if reset_all_axes
    for i = 1:length(h_all_axes)
        set(fig, 'CurrentAxes', h_all_axes(i));
        set(0,'CurrentFigure', fig);
        set(h_all_axes(i),'xlimmode', 'auto', 'ylimmode', 'auto');
        axis('image');
    end;
else
    %    set(h_all_axes(current_axes_index),'xlims', xlims(i,:), 'ylims',ylims(i,:));
    set(fig, 'CurrentAxes', last_axes);
    set(0,'CurrentFigure', fig);
    set(last_axes, 'xlimmode', 'auto', 'ylimmode', 'auto');
    axis('image');
end;
set(handlesPZ.Zoom_value_edit,'String', num2str(1)); 



%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Switch_Pan_Zoom;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% switch between active panning (default case) and active zooming via zoom on;
% only change to zooming is the fact that we decide to keep a square axis at 
% all times so an extra step to the windowbuttonupfcn is appended.
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

current_radiobutton = gcbo;

fig2 = findobj('Tag', 'PZ_figure');
handlesPZ = guidata(fig2);

Pan_On = get(handlesPZ.Pan_radiobutton, 'Value');
Zoom_On = get(handlesPZ.Zoom_radiobutton, 'Value');
userdata = get(handlesPZ.Reset_pushbutton, 'userdata');
fig = userdata{1};

if strcmp(get(current_radiobutton, 'Tag'), 'Pan_radiobutton')
    % pan button was last to be used
    if Pan_On
        Panning = 1;
    else
        Panning = 0;
    end;
else;
    % zoom button was last to be used
    if Zoom_On
        Panning = 0;
    else
        Panning = 1;
    end;
end;

Zooming = ~Panning;
set(handlesPZ.Pan_radiobutton, 'Value', Panning);
set(handlesPZ.Zoom_radiobutton, 'Value', Zooming);
set(0,'CurrentFigure', fig);
if Panning
    % restore mouse pointer to hand
    % restore WBDF and WBUF and WBMF by turning zoom off
    myzoom('off');
    
    % change the pointer to open hand
    hNewButton = findobj(gcf, 'Tag', 'figPanZoom');
    data = get(hNewButton,'Userdata');
    set(fig,'Pointer', 'Custom');
elseif Zooming
    % set the pointer to arrows
    set(fig,'Pointer', 'Arrow');    
    % turn zooming on;
    myzoom('on');
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Done_Zoom;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;
fig2 = findobj('Tag', 'PZ_figure');
handlesPZ = guidata(fig2);
old_axes_info = get(handlesPZ.Reset_pushbutton, 'UserData');
fig = old_axes_info{1};
figure(fig);
h_axes = gca;

im = get(findobj(h_axes, 'Type', 'image'), 'Cdata');

% Ta daa
axis('equal');
xlim = get(h_axes, 'Xlim');
ylim = get(h_axes, 'ylim');

zoom_factor_x = size(im,2) / diff(xlim);
zoom_factor_y = size(im,1) / diff(ylim);

set(handlesPZ.Zoom_value_edit, 'String', num2str(max([zoom_factor_x, zoom_factor_y]),3));

% now check if we need to apply new limits to all axes
apply_all = get(handlesPZ.Apply_radiobutton,'Value');
userdata = get(handlesPZ.Reset_pushbutton, 'UserData');
h_all_axes = userdata{3};
userdata{6} = h_axes;

if apply_all
    for i =1:length(h_all_axes);
        set(h_all_axes(i), 'xlim', xlim, 'ylim', ylim); 
    end;
end;
set(handlesPZ.Reset_pushbutton, 'UserData',userdata);

figure(fig2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Menu_Pan_Zoom;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

hNewMenu = gcbo;
checked=  umtoggle(hNewMenu);
hNewButton = get(hNewMenu, 'userdata');

if ~checked
    % turn off button
    %Deactivate_Pan_Zoom(hNewButton);
    set(hNewMenu, 'Checked', 'off');
    set(hNewButton, 'State', 'off' );
else
    %Activate_Pan_Zoom(hNewButton);
    set(hNewMenu, 'Checked', 'on');
    set(hNewButton, 'State', 'on' );
end;




%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Close_Parent_Figure;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to make sure that if parent figure is closed, 
% the tool figure is also closed
%global DB; if DB disp(['PZ_Tool: ', Get_Current_Function]); end;

set(findobj('Tag', 'PZ_figure'), 'Closerequestfcn', 'closereq');
try 
    delete(findobj('Tag','PZ_figure'));
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function  func_name = Get_Current_Function;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Debug function - returns current function name
x = dbstack;
x = x(2).name;
func_name = x(findstr('(', x)+1:findstr(')', x)-1);

% func_name = [];
% for i = length(x):-1:2
% 	if ~isempty(findstr('(', x(i).name))
% 		func_name = [func_name, x(i).name(findstr('(', x(i).name)+1:findstr(')', x(i).name)-1), ' : '];
% 	end;
% end;
% func_name = func_name(1:end-3);
