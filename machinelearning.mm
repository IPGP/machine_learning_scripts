<map version="freeplane 1.5.9">
<!--To view this file, download free mind mapping software Freeplane from http://freeplane.sourceforge.net -->
<node TEXT="machine learning" FOLDED="false" ID="ID_1287545166" CREATED="1489148131279" MODIFIED="1489150971749" STYLE="oval">
<font SIZE="18"/>
<hook NAME="MapStyle">
    <properties fit_to_viewport="false;"/>

<map_styles>
<stylenode LOCALIZED_TEXT="styles.root_node" STYLE="oval" UNIFORM_SHAPE="true" VGAP_QUANTITY="24.0 pt">
<font SIZE="24"/>
<stylenode LOCALIZED_TEXT="styles.predefined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="default" COLOR="#000000" STYLE="fork">
<font NAME="SansSerif" SIZE="10" BOLD="false" ITALIC="false"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.details"/>
<stylenode LOCALIZED_TEXT="defaultstyle.attributes">
<font SIZE="9"/>
</stylenode>
<stylenode LOCALIZED_TEXT="defaultstyle.note" COLOR="#000000" BACKGROUND_COLOR="#ffffff" TEXT_ALIGN="LEFT"/>
<stylenode LOCALIZED_TEXT="defaultstyle.floating">
<edge STYLE="hide_edge"/>
<cloud COLOR="#f0f0f0" SHAPE="ROUND_RECT"/>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.user-defined" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="styles.topic" COLOR="#18898b" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subtopic" COLOR="#cc3300" STYLE="fork">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.subsubtopic" COLOR="#669900">
<font NAME="Liberation Sans" SIZE="10" BOLD="true"/>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.important">
<icon BUILTIN="yes"/>
</stylenode>
</stylenode>
<stylenode LOCALIZED_TEXT="styles.AutomaticLayout" POSITION="right" STYLE="bubble">
<stylenode LOCALIZED_TEXT="AutomaticLayout.level.root" COLOR="#000000" STYLE="oval" SHAPE_HORIZONTAL_MARGIN="10.0 pt" SHAPE_VERTICAL_MARGIN="10.0 pt">
<font SIZE="18"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,1" COLOR="#0033ff">
<font SIZE="16"/>
<edge COLOR="#ff0000"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,2" COLOR="#00b439">
<font SIZE="14"/>
<edge COLOR="#0000ff"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,3" COLOR="#990000">
<font SIZE="12"/>
<edge COLOR="#00ff00"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,4" COLOR="#111111">
<font SIZE="10"/>
<edge COLOR="#ff00ff"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,5">
<edge COLOR="#00ffff"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,6">
<edge COLOR="#7c0000"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,7">
<edge COLOR="#00007c"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,8">
<edge COLOR="#007c00"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,9">
<edge COLOR="#7c007c"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,10">
<edge COLOR="#007c7c"/>
</stylenode>
<stylenode LOCALIZED_TEXT="AutomaticLayout.level,11">
<edge COLOR="#7c7c00"/>
</stylenode>
</stylenode>
</stylenode>
</map_styles>
</hook>
<hook NAME="AutomaticEdgeColor" COUNTER="10" RULE="ON_BRANCH_CREATION"/>
<hook NAME="accessories/plugins/AutomaticLayout.properties" VALUE="ALL"/>
<node TEXT="model" POSITION="right" ID="ID_1486859491" CREATED="1489148242353" MODIFIED="1489149116450" HGAP_QUANTITY="14.749999977648258 pt" VSHIFT_QUANTITY="-58.49999825656419 pt">
<edge COLOR="#00ffff"/>
<node TEXT="regression (fit Y = M(X))" ID="ID_848707408" CREATED="1489148691975" MODIFIED="1489150550042">
<node TEXT="nearest neighbor" ID="ID_192120717" CREATED="1489150509095" MODIFIED="1489150523538"/>
<node TEXT="decision tree" ID="ID_1981317805" CREATED="1489150524625" MODIFIED="1489150527641"/>
<node TEXT="Lasso regression" ID="ID_503979546" CREATED="1489150558163" MODIFIED="1489150564677"/>
<node TEXT="Kernel-Ridge Regression" ID="ID_1348172246" CREATED="1489150564887" MODIFIED="1489150597369"/>
</node>
<node TEXT="classification (grouping)" ID="ID_588018190" CREATED="1489148695582" MODIFIED="1489150359316">
<node TEXT="clustering (direction+amplitude)" ID="ID_1777013010" CREATED="1489149162379" MODIFIED="1489150111332">
<node TEXT="k-means" ID="ID_336421660" CREATED="1489149653234" MODIFIED="1489149655356"/>
<node TEXT="unsupervised" ID="ID_36202170" CREATED="1489149592027" MODIFIED="1489149594477"/>
</node>
<node TEXT="dimensionality reduction (direction)" ID="ID_1392975395" CREATED="1489149787988" MODIFIED="1489150101395">
<node TEXT="ICA" ID="ID_218016171" CREATED="1489149893947" MODIFIED="1489149895612"/>
<node TEXT="PCA" ID="ID_1245943003" CREATED="1489149895794" MODIFIED="1489149896797"/>
<node TEXT="Non-negative Matrix Factorization" ID="ID_1996598435" CREATED="1489149897138" MODIFIED="1489149909282"/>
<node TEXT="Kernel-PCA" ID="ID_1100254566" CREATED="1489150363316" MODIFIED="1489150365864"/>
</node>
<node TEXT="support vector machine (by label)" ID="ID_561689084" CREATED="1489149585582" MODIFIED="1489150290394">
<node TEXT="find decision boundary with largest separation" ID="ID_1290335787" CREATED="1489150295357" MODIFIED="1489150306301"/>
<node TEXT="supervised" ID="ID_1561153748" CREATED="1489149626389" MODIFIED="1489149629226"/>
</node>
</node>
</node>
<node TEXT="data" POSITION="left" ID="ID_792802981" CREATED="1489148269347" MODIFIED="1489148361912" HGAP_QUANTITY="51.499998882412946 pt" VSHIFT_QUANTITY="-32.99999901652339 pt">
<edge COLOR="#7c0000"/>
<node TEXT="design matrix (X)" ID="ID_1819002343" CREATED="1489148363355" MODIFIED="1489148426035">
<node TEXT="nsamples" ID="ID_1454062969" CREATED="1489148370226" MODIFIED="1489148373056"/>
<node TEXT="nfeatures" ID="ID_1587135104" CREATED="1489148373222" MODIFIED="1489148375647">
<node TEXT="amplitude" ID="ID_617289155" CREATED="1489149814162" MODIFIED="1489149816903"/>
<node TEXT="direction" ID="ID_1262564200" CREATED="1489149817069" MODIFIED="1489149818568"/>
</node>
</node>
<node TEXT="label (Y)" ID="ID_1202161884" CREATED="1489148397903" MODIFIED="1489148429994">
<node TEXT="nsamples" ID="ID_837785023" CREATED="1489148402342" MODIFIED="1489148404034"/>
</node>
</node>
<node TEXT="cost" POSITION="right" ID="ID_851161514" CREATED="1489148292327" MODIFIED="1489150969131" HGAP_QUANTITY="62.74999854713682 pt" VSHIFT_QUANTITY="-56.99999830126768 pt">
<edge COLOR="#00007c"/>
</node>
<node TEXT="optimization" POSITION="left" ID="ID_1943930913" CREATED="1489148298922" MODIFIED="1489148357145" HGAP_QUANTITY="9.500000134110447 pt" VSHIFT_QUANTITY="32.24999903887513 pt">
<edge COLOR="#007c00"/>
</node>
<node TEXT="examples" POSITION="right" ID="ID_1635363812" CREATED="1489150723402" MODIFIED="1489150971749" HGAP_QUANTITY="37.249999307096026 pt" VSHIFT_QUANTITY="-3.7499998882412946 pt">
<edge COLOR="#007c7c"/>
<node TEXT="classify spectrogram" ID="ID_521275785" CREATED="1489150742387" MODIFIED="1489150888492">
<node TEXT="dimensionality reduction" ID="ID_517410820" CREATED="1489150746750" MODIFIED="1489150757403">
<node TEXT="find particular spectral directions" ID="ID_1256857110" CREATED="1489150762018" MODIFIED="1489150779032"/>
</node>
<node TEXT="clustering" ID="ID_1605992119" CREATED="1489150782692" MODIFIED="1489150784268">
<node TEXT="normalized can find directions as well" ID="ID_384912511" CREATED="1489150784778" MODIFIED="1489150860105"/>
</node>
</node>
<node TEXT="spectrogram regression" ID="ID_85735668" CREATED="1489150893662" MODIFIED="1489150903778"/>
</node>
</node>
</map>
