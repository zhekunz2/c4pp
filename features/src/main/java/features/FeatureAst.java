import info.debatty.java.lsh.LSHSuperBit;
import org.antlr.v4.runtime.*;
import info.debatty.java.lsh.LSHMinHash;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import template.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class FeatureAst extends Template2BaseListener{
    public static HashMap<String, Integer> fixed_hash = createMap();
    public HashMap<String, Integer> feature_vector;
    public int parentLevel;
    public HashMap<String, Integer> hashes;
    public LSHSuperBit minHash;
    private static HashMap<String, Integer> createMap() {
        HashMap<String,Integer> myMap = new HashMap<String,Integer>();
        myMap.put("TernaryContext", 58);
        myMap.put("DataContext", 17);
        myMap.put("ParamsContext", 48);
        myMap.put("Function_declContext", 32);
        myMap.put("VecdivopContext", 64);
        myMap.put("StringContext", 55);
        myMap.put("Array_accessContext", 13);
        myMap.put("BracketsContext", 16);
        myMap.put("NeContext", 44);
        myMap.put("GtContext", 36);
        myMap.put("TemplateContext", 57);
        myMap.put("DimsContext", 19);
        myMap.put("LimitsContext", 39);
        myMap.put("VecmulopContext", 65);
        myMap.put("MulopContext", 43);
        myMap.put("LtContext", 41);
        myMap.put("AddopContext", 10);
        myMap.put("MinusopContext", 42);
        myMap.put("Return_or_param_typeContext", 53);
        myMap.put("DivopContext", 21);
        myMap.put("SubsetContext", 56);
        myMap.put("RefContext", 52);
        myMap.put("Function_callContext", 31);
        myMap.put("If_stmtContext", 37);
        myMap.put("FunctionContext", 30);
        myMap.put("TransformeddataContext", 59);
        myMap.put("ObserveContext", 46);
        myMap.put("PriorContext", 50);
        myMap.put("EqContext", 24);
        myMap.put("NumberContext", 45);
        myMap.put("ArrayContext", 12);
        myMap.put("DtypeContext", 22);
        myMap.put("FparamContext", 28);
        myMap.put("ParamContext", 47);
        myMap.put("LeqContext", 38);
        myMap.put("VectorDIMSContext", 67);
        myMap.put("TransformedparamContext", 60);
        myMap.put("QueryContext", 51);
        myMap.put("LoopcompContext", 40);
        myMap.put("BlockContext", 15);
        myMap.put("AssignContext", 14);
        myMap.put("FunctionsContext", 33);
        myMap.put("__class__", 68);
        myMap.put("UnaryContext", 62);
        myMap.put("FparamsContext", 29);
        myMap.put("ExprContext", 26);
        myMap.put("DistexprContext", 20);
        myMap.put("StatementContext", 54);
        myMap.put("AndContext", 11);
        myMap.put("GeneratedquantitiesContext", 34);
        myMap.put("For_loopContext", 27);
        myMap.put("TransposeContext", 61);
        myMap.put("DeclContext", 18);
        myMap.put("ValContext", 63);
        myMap.put("ExponopContext", 25);
        myMap.put("PrimitiveContext", 49);
        myMap.put("GeqContext", 35);
        myMap.put("Else_blkContext", 23);
        myMap.put("VectorContext", 66);
        return myMap;
    }


    public FeatureAst(int level){
        this.feature_vector = new HashMap<String, Integer>();
        this.parentLevel = level;
        this.hashes = new HashMap<>();
        this.minHash = new LSHSuperBit(5, 100, 2,98412194);
    }

    public String getHash(ArrayList<Integer> vector){
        int n = vector.size();
        if (n<this.parentLevel){
            for (int i = n; i < parentLevel; i++){
                vector.add(0);
            }
        }

        List<String> join = new ArrayList<>();
        for (int i = 0; i < vector.size(); i++){
            join.add(Integer.toString(vector.get(i)));
        }


        int[] h = new int[vector.size()];
        for (int i = 0; i < vector.size(); i++){
            h[i] = vector.get(i);
        }
        int[] hashed = this.minHash.hashSignature(h);
        List<String> newjoin = new ArrayList<>();
        for (int i = 0; i < hashed.length; i++){
            newjoin.add(Integer.toString(hashed[i]));
        }

//        String result = String.join("_", join);
        String result = String.join("_", newjoin);

        return result;
    }

    public ArrayList<Integer> getParents(ParserRuleContext ctx){
        int curLevel = 0;
        ParserRuleContext curNode = ctx;
        ArrayList<Integer> path = new ArrayList<Integer>();
        while (curNode!=null && curLevel < this.parentLevel){
            String nodeName = curNode.getClass().getSimpleName();
            path.add(fixed_hash.get(nodeName));
            curLevel+=1;
            curNode = curNode.getParent();
        }
        return path;
    }

    public void update_vector(ParserRuleContext ctx){
        String name;
        if (this.parentLevel<=1){
            name = ctx.getClass().getSimpleName();
            if (ctx.getParent()!=null){
                String parentName = ctx.getParent().getClass().getSimpleName();
                String feature_name = "t_" + parentName + "_" + name;
                if (!this.feature_vector.containsKey(feature_name)){
                    this.feature_vector.put(feature_name, 0);
                }
                int val = this.feature_vector.get(feature_name);
                this.feature_vector.put(feature_name, val+1);
            }
        }else {
            ArrayList<Integer> path = getParents(ctx);
            name = getHash(path);
        }
        if(!this.feature_vector.containsKey(name)){
            this.feature_vector.put(name, 0);
        }
        int val = this.feature_vector.get(name);
        this.feature_vector.put(name, val+1);
    }

    @Override
    public void enterAddop(Template2Parser.AddopContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterAnd(Template2Parser.AndContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterArray(Template2Parser.ArrayContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterArray_access(Template2Parser.Array_accessContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterAssign(Template2Parser.AssignContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterBlock(Template2Parser.BlockContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterBrackets(Template2Parser.BracketsContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterData(Template2Parser.DataContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterDecl(Template2Parser.DeclContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterPrimitive(Template2Parser.PrimitiveContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterNumber(Template2Parser.NumberContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterDtype(Template2Parser.DtypeContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterVector(Template2Parser.VectorContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterDims(Template2Parser.DimsContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterVectorDIMS(Template2Parser.VectorDIMSContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterLimits(Template2Parser.LimitsContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterPrior(Template2Parser.PriorContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterParam(Template2Parser.ParamContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterParams(Template2Parser.ParamsContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterDistexpr(Template2Parser.DistexprContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterLoopcomp(Template2Parser.LoopcompContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterFor_loop(Template2Parser.For_loopContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterIf_stmt(Template2Parser.If_stmtContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterElse_blk(Template2Parser.Else_blkContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterFunction_call(Template2Parser.Function_callContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterFparam(Template2Parser.FparamContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterFparams(Template2Parser.FparamsContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterReturn_or_param_type(Template2Parser.Return_or_param_typeContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterFunction_decl(Template2Parser.Function_declContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterTransformedparam(Template2Parser.TransformedparamContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterTransformeddata(Template2Parser.TransformeddataContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterGeneratedquantities(Template2Parser.GeneratedquantitiesContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterFunctions(Template2Parser.FunctionsContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterVal(Template2Parser.ValContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterDivop(Template2Parser.DivopContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterString(Template2Parser.StringContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterExponop(Template2Parser.ExponopContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterMinusop(Template2Parser.MinusopContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterLt(Template2Parser.LtContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterUnary(Template2Parser.UnaryContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterEq(Template2Parser.EqContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterGt(Template2Parser.GtContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterRef(Template2Parser.RefContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterGeq(Template2Parser.GeqContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterMulop(Template2Parser.MulopContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterFunction(Template2Parser.FunctionContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterVecmulop(Template2Parser.VecmulopContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterNe(Template2Parser.NeContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterLeq(Template2Parser.LeqContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterTranspose(Template2Parser.TransposeContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterVecdivop(Template2Parser.VecdivopContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterTernary(Template2Parser.TernaryContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterSubset(Template2Parser.SubsetContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterObserve(Template2Parser.ObserveContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterStatement(Template2Parser.StatementContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterQuery(Template2Parser.QueryContext ctx) { this.update_vector(ctx); }
    @Override
    public void enterTemplate(Template2Parser.TemplateContext ctx) { this.update_vector(ctx); }


    public static void update_table(HashMap<String, String> map, String output_path) throws Exception {
        File f = new File(output_path);
        f.getParentFile().mkdirs();
        f.createNewFile();
        try (PrintWriter writer = new PrintWriter(f)){
            StringBuilder keys = new StringBuilder();
            StringBuilder values = new StringBuilder();
            keys.append("program"+",");
            values.append(map.get("program")+",");
            for (String key: map.keySet()){
                if (!key.equals("program")) {
                    keys.append(key + ",");
                    values.append(map.get(key) + ",");
                }
            }
            keys.deleteCharAt(keys.length() - 1);
            keys.append("\n");
            values.deleteCharAt(values.length() - 1);
            values.append("\n");
            keys.append(values);
            System.out.println(keys);
            writer.write(keys.toString());
        } catch (Exception e){
            e.printStackTrace();
        }
    }


    public static void main(String[] args){
        long start = System.currentTimeMillis();
        CharStream stream = null;
        try {
            stream = CharStreams.fromFileName(args[0]);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Template2Lexer lexer = new Template2Lexer(stream);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        Template2Parser parser = new Template2Parser(tokens);
        int level = args.length>2? Integer.valueOf(args[2]):2;
        FeatureAst featureParser = new FeatureAst(level);
        Template2Parser.TemplateContext code = parser.template();
        ParseTreeWalker walker = new ParseTreeWalker();
        walker.walk(featureParser, code);
        File dir = new File(args[0].replace("ctemplate.template", ""));
        File stanfile=dir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String filename)
            { return filename.endsWith(".stan"); }
        } )[0];
        System.out.println(stanfile);
        HashMap<String, String> fv = new HashMap<>();
        for (String key: featureParser.feature_vector.keySet()) {
            fv.put(key, featureParser.feature_vector.get(key).toString());
        }
        fv.put("program", stanfile.toString());
        try {
            update_table(fv, args[1]);
            System.out.println("Size of keys: "+ fv.keySet().size());
        } catch (Exception e) {
            e.printStackTrace();
        }
        long end = System.currentTimeMillis();
        System.out.println(((end-start)/1000.0) + "s");

    }
}
