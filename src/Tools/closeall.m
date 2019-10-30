function closeall;
try 
	s = get(0,'ShowHiddenHandles');
	set(0,'ShowHiddenHandles', 'on');
	close all;
	set(0,'ShowHiddenHandles', s);
catch
	delete(gcf)
end