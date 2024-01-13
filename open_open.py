import subprocess

# 这行代码创建了一个字符串cmd，它包含了要执行的AppleScript命令。osascript是在命令行中运行AppleScript的工具。

# f'...': 这是Python的格式化字符串（f-string），允许在字符串中插入变量的值。
# osascript -e: 这是调用osascript工具并执行单条AppleScript命令的方式。
# \'activate application "{app_name}"\': 这是AppleScript命令，用于激活（或将焦点移至）指定的应用程序。使用转义字符\'来包含单引号内的字符串。

app_name = "QQ"
cmd = f'osascript -e \'activate application "{app_name}"\''

# subprocess.call: 这个函数用于执行指定的命令。在这个例子中，它执行的是cmd字符串中的AppleScript命令。
# shell=True: 这个参数允许命令在shell（如bash或zsh）中执行，这是运行复杂命令所必需的。
# 在macOS中，标准的应用程序通常安装在 /Applications 目录下。
# 使用 AppleScript 激活或打开应用程序时，你通常不需要指定应用程序的完整路径。
# 此脚本会告诉 AppleScript 去寻找一个名为 "Photoshop" 的应用程序并将其激活。AppleScript 会在标准的应用程序目录中查找匹配的应用程序。

subprocess.call(cmd, shell=True)

##########################
# Alternative of opening app
##########################
import subprocess

# 在 macOS 中使用 subprocess 来打开应用程序时，应当使用 open -a 命令，而不是直接指定应用程序的名称。open -a 命令是 macOS 特有的，用于打开应用程序。
# 同样也是直接在标准的应用程序目录中查找匹配的应用程序。
subprocess.Popen(['open', '-a', 'Notes'])


