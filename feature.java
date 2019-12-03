package features;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.json.simple.JSONObject;
import translator.*;

import java.io.*;
import java.util.*;

public class feature extends StanBaseListener{
    public HashMap<String, Integer> feature_vector;
    public String cur_block;
    public ArrayList<String> data_items;
    public int curr_loop_deg;
    public String cur_loop_count;
    public ArrayList<ParserRuleContext> stack;
    public ArrayList<String> parameters;
    public String sink;
    public int arithComplexity;
    public ArrayList<String> observe_variables;
    public int curr_data_dim;
    public int curr_param_dim;
    public static HashMap<String, String> disc_dict = createDisc();

    public static HashMap<String, String> createDisc(){
        HashMap<String, String> res = new HashMap<>();
        res.put("neg_binomial", "disc");
        res.put("binomial", "disc");
        res.put("bernoulli", "disc");
        res.put("categorical", "disc");
        res.put("hypergeometric", "disc");
        return res;
    }

    public feature(){
        this.feature_vector = new HashMap<String, Integer>();
        this.cur_block = "";
        this.data_items = new ArrayList<>();
        this.curr_loop_deg = 0;
        this.cur_loop_count = null;
        this.stack = new ArrayList<>();
        this.parameters = new ArrayList<>();
        StanParser.ProgramContext ctx = new StanParser.ProgramContext(null, 0);
        this.stack.add(ctx);
        this.sink = null;
        this.arithComplexity = 0;
        this.observe_variables = new ArrayList<>();
    }

    public void updateVector(String key){
        if (this.feature_vector.containsKey(key)){
            feature_vector.put(key, feature_vector.get(key)+1);
        } else {
            feature_vector.put(key, 1);
        }
    }

    public void updateVectorMax(String key, Integer cur){
        if (!this.feature_vector.containsKey(key)){
            feature_vector.put(key, cur);
        } else if (feature_vector.get(key)<cur){
            feature_vector.put(key, cur);
        }
    }

    @Override
    public void enterDatablk(StanParser.DatablkContext ctx){
        this.cur_block = "data";
    }
    @Override
    public void exitDatablk(StanParser.DatablkContext ctx){
        this.cur_block = "";
    }

    @Override
    public void enterDecl(StanParser.DeclContext ctx){
        String id = ctx.ID().getText();
        if (this.cur_block.equals("data")){
            String data_type = ctx.type().getText();
            updateVector("d_"+data_type);
            this.curr_data_dim = 0;
            this.data_items.add(id);
        } else if(cur_block.equals("parameters")){
            updateVector("param");
            String param_type = ctx.type().getText();
            updateVector(param_type);
            this.curr_data_dim = 0;
            this.parameters.add(id);
        } else if (this.cur_block.equals("model")){
            updateVector("internal");
        }
    }

    @Override
    public void exitDecl(StanParser.DeclContext ctx){
        if (cur_block.equals("data")){
            if (feature_vector.containsKey("max_data_dim")){
                feature_vector.put("max_data_dim", Math.max(feature_vector.get("max_data_dim"), curr_data_dim));
            } else {
                feature_vector.put("max_data_dim", curr_data_dim);
            }
            curr_data_dim = 0;
        } else if (cur_block.equals("parameters")){
            if (feature_vector.containsKey("max_param_dim")){
                feature_vector.put("max_param_dim", Math.max(feature_vector.get("max_param_dim"), curr_param_dim));
            } else {
                feature_vector.put("max_param_dim", curr_param_dim);
            }
            curr_param_dim = 0;
        }
    }

    @Override
    public void enterDims(StanParser.DimsContext ctx){
        if (cur_block.equals("data")){
            curr_data_dim += ctx.dim().size();
        }
        if (cur_block.equals("parameters")){
            curr_param_dim += ctx.dim().size();
        }
    }

    @Override
    public void enterFunction_call(StanParser.Function_callContext ctx){
        stack.add(ctx);
        updateVector("f_"+ctx.inbuilt().getText());
    }

    @Override
    public void exitFunction_call(StanParser.Function_callContext ctx){
        stack.remove(stack.size()-1);
    }

    @Override
    public void enterDistribution_exp(StanParser.Distribution_expContext ctx){
        String dist = ctx.ID().getText();
        if (dist.equals("dirichlet")){
            if (cur_loop_count != null){
                feature_vector.put("categories", Integer.valueOf(cur_loop_count));
            }
        }
        stack.add(ctx);
        updateVector(dist);
        if (disc_dict.containsKey(dist)){
            updateVector("disc");
        }
        if (sink != null && data_items.contains(sink)){
            updateVector("d_"+dist);
            updateVector("observe");
            observe_variables.add(sink);
        }
    }

    @Override
    public void exitDistribution_exp(StanParser.Distribution_expContext ctx){
        stack.remove(stack.size()-1);
        String dist = ctx.ID().getText();
        if (sink!=null && !data_items.contains(sink)){
            // TODO: implement this eval() in java.
            ArrayList<StanParser.ExpressionContext> dist_param = new ArrayList<>();
            for (StanParser.ExpressionContext ee : ctx.expression()){
                System.out.println(ee.getText());
            }
        }
    }

    @Override
    public void enterId_access(StanParser.Id_accessContext ctx){
        String id = ctx.ID().getText();
        if ((stack.get(stack.size()-1) instanceof StanParser.Distribution_expContext)&& parameters.contains(id)){
            if (sink != null && parameters.contains(sink)){
                updateVector("dependent_prior");
            }
        }
    }

    @Override
    public void enterArray_access(StanParser.Array_accessContext ctx){
        String id = ctx.ID().getText();
        if ((stack.get(stack.size()-1) instanceof StanParser.Distribution_expContext)&& parameters.contains(id)){
            if (sink != null && parameters.contains(sink)){
                updateVector("dependent_prior");
            }
        }
    }

    @Override
    public void enterModelblk(StanParser.ModelblkContext ctx){ cur_block = "model"; }

    @Override
    public void exitModelblk(StanParser.ModelblkContext ctx){ cur_block = ""; }

    @Override
    public void enterParamblk(StanParser.ParamblkContext ctx){ cur_block = "parameters"; }

    @Override
    public void exitParamblk(StanParser.ParamblkContext ctx){ cur_block = ""; }

    @Override
    public void enterFor_loop_stmt(StanParser.For_loop_stmtContext ctx){
        updateVector("loop");
        curr_loop_deg +=1;
        String loopindex = ctx.range_exp().expression(1).getText();
        if (data_items.contains(loopindex)){
            cur_loop_count = loopindex;
        }
        if (feature_vector.containsKey("loop_deg")){
            feature_vector.put("loop_deg", Math.max(feature_vector.get("loop_deg"), curr_loop_deg));
        } else {
            feature_vector.put("loop_deg", 1);
        }
    }

    @Override
    public void exitFor_loop_stmt(StanParser.For_loop_stmtContext ctx){
        curr_loop_deg --;
        cur_loop_count = null;
    }

    @Override
    public void enterIf_stmt(StanParser.If_stmtContext ctx){
        updateVector("if");
    }

    @Override
    public void enterTernary_if(StanParser.Ternary_ifContext ctx){
        updateVector("ternary_if");
    }

    @Override
    public void enterSample(StanParser.SampleContext ctx){
        updateVector("sample");
        StanParser.ExpressionContext lhs = ctx.expression();
        if (lhs instanceof StanParser.Id_accessContext){
            sink = ((StanParser.Id_accessContext) lhs).ID().getText();
        } else if (lhs instanceof StanParser.ArrayContext){
            sink = ((StanParser.ArrayContext) lhs).array_access().ID().getText();
        } else if (lhs instanceof StanParser.FunctionContext){
            if (((StanParser.FunctionContext) lhs).function_call().inbuilt().getText().equals("to_vector")){
                sink = ((StanParser.FunctionContext) lhs).function_call().expression(0).getText();
            } else {
                System.out.println(lhs.getText());
                return;
            }
        } else {
            System.out.println(ctx.getText());
            assert false;
        }
    }

    @Override
    public void enterExponop(StanParser.ExponopContext ctx){
        arithComplexity ++;
        updateVectorMax("arith", arithComplexity);
        updateVector("arithops");
    }

    @Override
    public void exitExponop(StanParser.ExponopContext ctx){
        arithComplexity --;
    }

    @Override
    public void enterDivop(StanParser.DivopContext ctx){
        arithComplexity ++;
        updateVectorMax("arith", arithComplexity);
        updateVector("arithops");
    }

    @Override
    public void exitDivop(StanParser.DivopContext ctx){
        arithComplexity --;
    }

    @Override
    public void enterMulop(StanParser.MulopContext ctx){
        arithComplexity ++;
        updateVectorMax("arith", arithComplexity);
        updateVector("arithops");
    }

    @Override
    public void exitMulop(StanParser.MulopContext ctx){
        arithComplexity --;
    }

    @Override
    public void enterAddop(StanParser.AddopContext ctx){
        arithComplexity ++;
        updateVectorMax("arith", arithComplexity);
        updateVector("arithops");
    }

    @Override
    public void exitAddop(StanParser.AddopContext ctx){
        arithComplexity --;
    }

    @Override
    public void enterMinusop(StanParser.MinusopContext ctx){
        arithComplexity ++;
        updateVectorMax("arith", arithComplexity);
        updateVector("arithops");
    }

    @Override
    public void exitMinusop(StanParser.MinusopContext ctx){
        arithComplexity --;
    }


    public static void main(String[] args) throws IOException {
        long start = System.currentTimeMillis();
        List<String> exception = new ArrayList<>();
        exception.add("model10-4");
        exception.add("model12-1");
//        exception.add("model10-5");
        File file = new File("/Users/zhekunz2/Desktop/SixthSense/example-models/time_series");
        String[] directories = file.list(new FilenameFilter() {
            @Override
            public boolean accept(File current, String name) {
                return new File(current, name).isDirectory();
            }
        });

        Arrays.sort(directories);
        for (String name : directories){
            if (!exception.contains(name)){
                System.out.println("Feature Parsing: "+name);
                String data_file = "/Users/zhekunz2/Desktop/SixthSense/example-models/time_series/"+name+"/"+name+".data.R";
                String stan_file_path = "/Users/zhekunz2/Desktop/SixthSense/example-models/time_series/"+name+"/"+name+".stan";
                CharStream stream = null;
                try {
                    stream = CharStreams.fromFileName(stan_file_path);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                StanLexer lexer = new StanLexer(stream);
                CommonTokenStream tokens = new CommonTokenStream(lexer);
                StanParser parser = new StanParser(tokens);
                feature featureParser = new feature();
                StanParser.ProgramContext code = parser.program();
                ParseTreeWalker walker = new ParseTreeWalker();
                walker.walk(featureParser, code);
                Map<String, Integer> fv = featureParser.feature_vector;
                Map<String, String> jsonmap = new HashMap<>();
                for (String key: fv.keySet()) {
                    jsonmap.put(key, fv.get(key).toString());
                }
                //write fv to json
                JSONObject fv_json = new JSONObject();
                fv_json.putAll( jsonmap );
                try (FileWriter fv_file = new FileWriter("output/"+name+"_fv.json")) {
                    fv_file.write(fv_json.toJSONString());
                    fv_file.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                //write observe_variable to file
                if (featureParser.observe_variables.size()!=0) {
                    PrintStream fileStream = new PrintStream(new File("output/"+name+"_obv.txt"));
                    for (String x : featureParser.observe_variables) {
                        fileStream.println(x);
                    }
                }
                String fv_path = "/Users/zhekunz2/Desktop/SixthSense/Translator/output/"+name+"_fv.json";
                String obv_path = "/Users/zhekunz2/Desktop/SixthSense/Translator/output/"+name+"_obv.txt";

                Process p = Runtime.getRuntime().exec("python3 /Users/zhekunz2/Desktop/SixthSense/c4pp/feature.py "+data_file+" "+fv_path+" "+obv_path);
                System.out.println("python3 ~/Desktop/SixthSense/c4pp/feature.py "+data_file+" "+fv_path+" "+obv_path);
            }
        }
        long end = System.currentTimeMillis();
        System.out.println(((end-start)/1000.0) + "s");

    }
}
