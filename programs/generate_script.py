import sys

name = sys.argv[1]
final_str = ""
final_str += "modelCheck(\'" + name + "model.txt\')\n"
final_str += "modelData(\'" + name + "data.txt\')\n"
final_str += "modelCompile(1)\n"
final_str += "modelInits(\'" + name + "inits.txt\')\n"
final_str += "\n"
final_str += "modelUpdate(1000)\n"
#with open(sys.argv[2] + '/bug_code/' + sys.argv[1] + 'inits.txt')
final_str += "samplesSet('theta')\n"
final_str += "samplesSet('X')\n"
final_str += "modelUpdate(100)\n"
final_str += "samplesStats('*')\n"

print(final_str)





