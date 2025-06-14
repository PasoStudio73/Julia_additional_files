using PkgTemplates

t = Template(
    user="PasoStudio73", 
    authors=["Giovanni Pagliarini", "pglgnn@unife.it", "Riccardo Pasini", "riccardo01.pasini@unife.it"],
    julia=v"1",
    plugins=[ProjectFile(; version=v"0.0.1")]
)
t("SoleXplorer.jl")


