function plan = buildfile
import matlab.buildtool.tasks.*

plan = buildplan(localfunctions);

plan("clean") = CleanTask;
plan("check") = CodeIssuesTask;
plan("test") = TestTask;

% Make the "archive" task the default task in the plan
plan.DefaultTasks = "package";

% Make the "archive" task dependent
plan("package").Dependencies = ["check" "clean" "test"];
end

function packageTask(~)
    % Define actions for packaging the toolbox
    disp("Packaging toolbox...");
    matlab.addons.toolbox.packageToolbox("mvBayes.prj");
end
